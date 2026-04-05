import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { registerStep } from '../engine.js';
import type { StepContext, StepResult } from '../types.js';
import type { Logger } from '../../utils/logger.js';
import { askAIJSON } from '../../utils/ai-client.js';
import type { AIBrain, AIUsage } from '../../utils/ai-client.js';
import { loadConfig } from '../../utils/config.js';

const execFileAsync = promisify(execFile);

// ── Types ────────────────────────────────────────

interface Word {
  id: number;
  word: string;
  start: number;
  end: number;
  is_presenter: boolean;
  confidence: number;
  final_decision?: 'presenter' | 'reject' | 'uncertain';
  final_score?: number;
  vad_score?: number;
  visual_score?: number;
  speaker_score?: number;
  asr_score?: number;
  energy_score?: number;
  take_id?: number;
}

interface MergedTranscript {
  words: Word[];
  stats: {
    total_words: number;
    presenter_words: number;
    other_words: number;
    filter_ratio: number;
    processing_time_ms: number;
  };
}

interface KeepRange {
  ids: number[];
  reason: string;
}

interface TakeDecision {
  ids: number[];
  reason: string;
  kept_ids?: number[];
}

interface TakeCandidateAlternative {
  ids: number[];
  score: number;
}

interface TakeCandidate {
  sentence_group: string;
  best_take_ids: number[];
  alternatives?: TakeCandidateAlternative[];
}

interface DuplicateVersion {
  take_idx: number;
  word_ids: number[];
  score: number;
  text: string;
  speaker_avg: number;
}

interface DuplicateCandidate {
  phrase_text: string;
  similarity: number;
  versions: DuplicateVersion[];
}

interface TakeDecisions {
  remove_ids: number[];
  decisions: TakeDecision[];
  candidates?: TakeCandidate[];
  duplicate_candidates?: DuplicateCandidate[];
  stats: {
    duplicates_found: number;
    false_starts: number;
    production_cues: number;
    stutters: number;
    abandoned_takes: number;
    hard_rejections?: number;
    total_removed_words: number;
  };
  processing_time_ms: number;
}

interface AICleanResult {
  keep_ranges: KeepRange[];
}

function buildTakeSelectorHints(takeDecisions: TakeDecisions): string {
  if (!takeDecisions || takeDecisions.decisions.length === 0) {
    return 'No automated issues detected.';
  }

  const lines: string[] = [];
  for (const d of takeDecisions.decisions) {
    const ids = d.ids.join(', ');
    switch (d.reason) {
      case 'production_cue':
        lines.push(`- Production instruction detected at words [${ids}] — likely "סליחה", "עוד פעם", etc.`);
        break;
      case 'duplicate_take':
        lines.push(`- Duplicate take detected: words [${ids}] are a repeat. Better version at words [${d.kept_ids?.join(', ')}]`);
        break;
      case 'false_start':
        lines.push(`- False start detected at words [${ids}] — presenter started and restarted`);
        break;
      case 'stutter':
        lines.push(`- Stutter detected at words [${ids}]`);
        break;
      case 'internal_retake':
        lines.push(`- Internal retake within a take at words [${ids}] — repeated phrase`);
        break;
      case 'abandoned_take':
        lines.push(`- Abandoned take at words [${ids}] — presenter gave up on this version`);
        break;
      case 'hard_reject':
        lines.push(`- Hard rejected words [${ids}] — failed multiple quality checks (DO NOT use these words)`);
        break;
    }
  }
  return lines.join('\n');
}

function buildCandidatesText(takeDecisions: TakeDecisions): string {
  if (!takeDecisions.candidates || takeDecisions.candidates.length === 0) {
    return 'No ranked candidates available.';
  }

  return takeDecisions.candidates.map((c, i) => {
    const bestIds = c.best_take_ids.join(',');
    const alts = c.alternatives?.map(a =>
      `  Alternative: IDs [${a.ids.join(',')}] (score: ${a.score.toFixed(2)})`
    ).join('\n') || '';
    return `Sentence group ${i + 1}: "${c.sentence_group}"
  BEST: IDs [${bestIds}]
${alts}`;
  }).join('\n\n');
}

interface KeepSegment {
  start: number;
  end: number;
  word_ids: number[];
}

interface RemoveSegment {
  start: number;
  end: number;
  reason: string;
}

// ── Helpers ──────────────────────────────────────

function readStatusFile(statusPath: string): Record<string, unknown> {
  if (existsSync(statusPath)) {
    try {
      return JSON.parse(readFileSync(statusPath, 'utf-8')) as Record<string, unknown>;
    } catch {
      return {};
    }
  }
  return {};
}

function writeStatus(statusPath: string, update: Record<string, unknown>): void {
  const current = readStatusFile(statusPath);
  const merged = { ...current, ...update };
  writeFileSync(statusPath, JSON.stringify(merged, null, 2), 'utf-8');
}

// ── Step 1: Merge (Python) ──────────────────────

async function runMerge(
  jobDir: string,
  logger: Logger,
): Promise<MergedTranscript> {
  const transcriptPath = join(jobDir, 'transcript.json');
  const segmentsPath = join(jobDir, 'presenter_segments.json');
  const outputPath = join(jobDir, 'merged_transcript.json');

  if (!existsSync(transcriptPath)) {
    throw new Error(`Transcript not found: ${transcriptPath}`);
  }

  const scriptPath = resolve('python', 'merge_transcript.py');

  // Determine python path from config
  const config = loadConfig();
  const pdConfig = (config as unknown as Record<string, unknown>).presenterDetection as
    { pythonPath?: string } | undefined;
  const pythonPath = pdConfig?.pythonPath ?? 'python3';

  const gatedAudioPath = join(jobDir, 'audio_gated.wav');
  const audioPath = existsSync(gatedAudioPath) ? gatedAudioPath : join(jobDir, 'audio.wav');
  const speakerVerify = (pdConfig as { speakerVerify?: boolean } | undefined)?.speakerVerify ?? false;

  const args = [
    scriptPath,
    '--transcript', transcriptPath,
    '--segments', segmentsPath,
    '--output', outputPath,
    '--buffer', '0.25',
    '--audio', audioPath,
    ...(speakerVerify ? ['--speaker-verify'] : []),
  ];

  logger.info('Running merge_transcript.py', { transcriptPath, segmentsPath, audioPath, speakerVerify });

  const { stdout, stderr } = await execFileAsync(pythonPath, args, {
    timeout: speakerVerify ? 600000 : 30000,
  });

  if (stderr) {
    logger.debug('merge_transcript.py stderr', { stderr: stderr.trim() });
  }

  // Parse result from stdout
  const result = JSON.parse(stdout) as MergedTranscript;

  logger.info('Merge completed', {
    totalWords: result.stats.total_words,
    presenterWords: result.stats.presenter_words,
    otherWords: result.stats.other_words,
    filterRatio: result.stats.filter_ratio,
  });

  return result;
}

// ── Step 1.25: WhisperX alignment (Python) ──

async function runAlignWords(
  jobDir: string,
  logger: Logger,
): Promise<MergedTranscript> {
  const mergedPath = join(jobDir, 'merged_transcript.json');
  const gatedAudioPath = join(jobDir, 'audio_gated.wav');
  const audioPath = existsSync(gatedAudioPath) ? gatedAudioPath : join(jobDir, 'audio.wav');
  const alignedPath = join(jobDir, 'merged_transcript.json'); // overwrite in place

  const scriptPath = resolve('python', 'align_words.py');

  const config = loadConfig();
  const pdConfig = (config as unknown as Record<string, unknown>).presenterDetection as
    { pythonPath?: string } | undefined;
  const pythonPath = pdConfig?.pythonPath ?? 'python3';

  const args = [
    scriptPath,
    '--audio', audioPath,
    '--words', mergedPath,
    '--output', alignedPath,
    '--language', 'he',
  ];

  logger.info('Running align_words.py (WhisperX forced alignment)', { audioPath });

  try {
    const { stdout, stderr } = await execFileAsync(pythonPath, args, {
      timeout: 600000,
    });

    if (stderr) {
      logger.debug('align_words.py stderr', { stderr: stderr.trim() });
    }

    const result = JSON.parse(stdout) as MergedTranscript;
    logger.info('WhisperX alignment completed', {
      method: (result.stats as Record<string, unknown>).alignment
        ? ((result.stats as Record<string, unknown>).alignment as Record<string, unknown>).method
        : 'unknown',
    });

    return result;
  } catch (err) {
    logger.warn('WhisperX alignment failed, using original timestamps', {
      error: (err as Error).message,
    });
    // Fallback: read the existing merged transcript
    return JSON.parse(readFileSync(mergedPath, 'utf-8')) as MergedTranscript;
  }
}

// ── Step 1.5: Take Selector (Python, rule-based) ──

async function runTakeSelector(
  jobDir: string,
  logger: Logger,
): Promise<TakeDecisions> {
  const mergedPath = join(jobDir, 'merged_transcript.json');
  const gatedAudioPath = join(jobDir, 'audio_gated.wav');
  const audioPath = existsSync(gatedAudioPath) ? gatedAudioPath : join(jobDir, 'audio.wav');
  const outputPath = join(jobDir, 'take_decisions.json');

  const scriptPath = resolve('python', 'take_selector.py');

  const config = loadConfig();
  const pdConfig = (config as unknown as Record<string, unknown>).presenterDetection as
    { pythonPath?: string } | undefined;
  const pythonPath = pdConfig?.pythonPath ?? 'python3';

  const tsConfig = (config as unknown as Record<string, unknown>).takeSelector as
    { enabled?: boolean; similarityThreshold?: number; lookbackSeconds?: number; scoringOverrideMargin?: number } | undefined;

  if (tsConfig?.enabled === false) {
    logger.info('Take selector disabled in config, skipping');
    return { remove_ids: [], decisions: [], stats: { duplicates_found: 0, false_starts: 0, production_cues: 0, stutters: 0, abandoned_takes: 0, total_removed_words: 0 }, processing_time_ms: 0 };
  }

  const args = [
    scriptPath,
    '--words', mergedPath,
    '--output', outputPath,
    '--video-type', 'general',
  ];

  if (existsSync(audioPath)) {
    args.push('--audio', audioPath);
  }

  if (tsConfig?.similarityThreshold !== undefined) {
    args.push('--similarity-threshold', String(tsConfig.similarityThreshold));
  }
  if (tsConfig?.lookbackSeconds !== undefined) {
    args.push('--lookback-seconds', String(tsConfig.lookbackSeconds));
  }
  if (tsConfig?.scoringOverrideMargin !== undefined) {
    args.push('--scoring-override-margin', String(tsConfig.scoringOverrideMargin));
  }

  logger.info('Running take_selector.py', { mergedPath, audioPath });

  const { stdout, stderr } = await execFileAsync(pythonPath, args, {
    timeout: 600000,
  });

  if (stderr) {
    logger.debug('take_selector.py stderr', { stderr: stderr.trim() });
  }

  const result = JSON.parse(stdout) as TakeDecisions;

  logger.info(`Take selector: ${result.stats.total_removed_words} words marked for removal (${result.stats.duplicates_found} duplicates, ${result.stats.false_starts} false_starts, ${result.stats.production_cues} cues, ${result.stats.stutters} stutters, ${result.stats.abandoned_takes} abandoned)`);

  return result;
}

// ── Step 2: Semantic cleanup (AI) ───────────────

function buildNumberedText(words: Word[], takeSelectorIds?: Set<number>): string {
  return words.map((w) => {
    if (takeSelectorIds?.has(w.id)) {
      return `~[${w.id}] ${w.word}~`;
    }
    return w.is_presenter ? `[${w.id}] ${w.word}` : `*[${w.id}] ${w.word}*`;
  }).join(' ');
}

function buildNumberedTextRaw(words: Word[]): string {
  return words.map((w) => `[${w.id}] ${w.word}`).join(' ');
}

function buildStructuredText(
  words: Word[],
  takeSelectorIds: Set<number>,
  hardRejectedIds: Set<number>,
  logger?: Logger,
): string {
  // Validate chronological order
  for (let i = 1; i < words.length; i++) {
    if (words[i]!.start < words[i - 1]!.start - 0.1) {
      logger?.warn('Words not in chronological order', {
        id: words[i]!.id, start: words[i]!.start,
        prevId: words[i - 1]!.id, prevStart: words[i - 1]!.start,
      });
    }
  }

  const parts: string[] = [];
  let takeNumber = 1;

  // Start with TAKE 1 header
  parts.push(`--- TAKE ${takeNumber} ---\n`);

  for (let i = 0; i < words.length; i++) {
    const w = words[i]!;

    // Skip hard-rejected words entirely — AI never sees them
    if (hardRejectedIds.has(w.id)) continue;

    let prefix = '';
    let suffix = '';

    if (!w.is_presenter && w.final_decision !== 'uncertain') {
      prefix = '*';
      suffix = '*';
    } else if (w.final_decision === 'uncertain') {
      prefix = '?';
      suffix = '?';
    } else if (takeSelectorIds.has(w.id)) {
      prefix = '~';
      suffix = '~';
    }

    parts.push(`${prefix}[${w.id}] ${w.word}${suffix}`);

    // GAP marker between takes — use take_id if available, fallback to time gap
    if (i < words.length - 1) {
      const next = words[i + 1]!;
      const gap = next.start - w.end;
      const differentTake = w.take_id !== undefined && next.take_id !== undefined
        && w.take_id !== next.take_id;

      if (differentTake) {
        takeNumber++;
        parts.push(`\n\n--- TAKE ${takeNumber} (${gap.toFixed(1)}s gap) ---\n`);
      }
    }
  }

  return parts.join(' ');
}

// ── AI Step 1: Narrative selection from ALL words ──

const NARRATIVE_PROMPT = `You are the world's best direct-response copywriter AND video editor combined.

You receive the COMPLETE transcript of a raw marketing video recording.
The presenter recorded multiple takes of the same script. Your job:
BUILD THE MOST COMPELLING MARKETING VIDEO POSSIBLE by selecting the
strongest, clearest, most complete version of each part of the message.

STEP 1 — READ AND MAP THE FULL MESSAGE:
Read the entire transcript. Identify EVERY unique marketing idea/sentence.
Map the complete sales structure:
- HOOK: The opening that grabs attention (pain point, question, shocking stat)
- PROBLEM: What the viewer is struggling with
- SOLUTION: What you offer and how it works
- MECHANISM: The specific process/method
- PROOF/GUARANTEE: Why they should trust you
- CTA: What to do next

DO NOT skip any part. A complete marketing video needs ALL parts.

STEP 2 — FOR EACH PART, FIND THE BEST TAKE:
The presenter said each part multiple times. For each part:
- Find ALL versions in the transcript
- Select the LAST (highest take number) complete version
- Make sure it's a COMPLETE sentence (starts and ends properly)

STEP 3 — BUILD THE FULL VIDEO:
Select segments that cover the ENTIRE marketing message.
MOST IMPORTANT RULE — MINIMUM COVERAGE:
Count the presenter words available (non-starred, non-flagged).
You MUST select at least 55% of them. If your selection is below
55% — you are cutting too much content. Go back and add more
segments until you reach at least 55%.
Count: if there are 170 presenter words, select at least 94.
If you're selecting less than 50% — you're cutting WAY too much!

MISSING PARTS CHECK: Before returning, verify you have:
✓ Hook (opening)
✓ Problem statement
✓ Solution/what you offer
✓ How it works (mechanism/process)
✓ Result/proof/guarantee
✓ Call to action
If ANY part is missing — find it in the transcript and add it!

A 30-40 second video from 2.5 minutes of raw footage should have
7-10 segments covering every part of the message.

CRITICAL RULES:

- STARRED WORDS — STRONG PREFERENCE:
  STRONGLY AVOID segments with 2+ consecutive starred words (*).
  Starred words are background voices — including them means the
  production assistant's voice will be heard in the final video.
  If no clean alternative exists, you MAY include them as last resort,
  but add 'starred_warning' to the reason field.
  A single isolated starred word between presenter words
  may be included if it completes the sentence.

- CHRONOLOGICAL ORDER: Selected segments MUST appear in the same order
  as they were recorded. If segment A starts at second 20 and segment B
  starts at second 50, A must come before B in your output.
  NEVER put a later timestamp before an earlier one.

- COMPLETE SENTENCES — CRITICAL:
  Every segment MUST be a sentence that makes sense ON ITS OWN.

  Test: Read just the segment text. Does it sound like a complete thought?

  BAD starts (fragments — NEVER start a segment with these):
  ✗ "שלך הוא לא..." (missing "העסק")
  ✗ "חד פעמי..." (missing "זה פרויקט")
  ✗ "לוקחים 23..." (missing "אנחנו")
  ✗ "שהחיסכון..." (starts with ש' prefix)
  ✗ "לך שהחיסכון..." (starts mid-sentence)

  FORBIDDEN FIRST WORDS — NEVER start a segment with any of these
  Hebrew connector/function words (they are ALWAYS mid-sentence):
  זמן, וכסף, ידנית, שלך, לך, שהחיסכון, פרטים, בחינם, מחליף,
  והופכים, אותם, ולא, ובלי, שגונבים, ומחכה, ואם, שיהיה
  If your segment starts with one of these — extend backward to include
  the sentence start, or remove the segment entirely.

  BAD ends (fragments — NEVER end a segment with these):
  ✗ "...המערכת לא" (hanging "לא")
  ✗ "...מחליף את" (hanging "את")

  GOOD segments (complete thoughts):
  ✓ "אם כל לקוח חדש אצלך גורם לך לחשוב אולי אני צריך עוד עובד"
  ✓ "העסק שלך הוא לא בנוי לצמיחה הוא בנוי לעבודה ידנית"
  ✓ "אנחנו לוקחים 2-3 תהליכים שגונבים לך הכי הרבה זמן"

- FULL MESSAGE COVERAGE: Your selection must include:
  * A hook (the opening)
  * The problem statement
  * The solution/what you offer
  * How it works (mechanism)
  * Proof or guarantee
  * Call to action
  If ANY of these is missing — go back and find it in the transcript.

- INCLUDE EVERYTHING: Every unique idea or sentence the presenter said
  should be represented in your selection. Do NOT skip content just because
  it's short or seems less important. The only reasons to skip a sentence are:
  (a) It's a duplicate of something you already selected
  (b) It's heavily stuttered/mumbled and no clean version exists
  (c) It's a production instruction ("סליחה", "עוד פעם", "סיימתי")
  If the presenter said it clearly at least once — it belongs in the video.
  The starred words preference still applies — avoid segments with 2+ starred words.
  Include everything the PRESENTER said, not everything in the transcript.

- TAKE BREAKS: Never select IDs that cross a "--- TAKE N ---" boundary.

- UNCERTAIN WORDS (?): Words marked with ? are UNCERTAIN — the system
  couldn't determine if they belong to the presenter or to background voice.
  Use your judgment: if the word fits naturally in a presenter sentence,
  include it. If it seems out of place — exclude it.

- PREFER CLEAN WORDS: Avoid words marked with * (non-presenter)
  and ~ (flagged by analysis). Strongly avoid including
  2+ consecutive starred words in a segment.

- LAST TAKE IS BEST: When the same sentence appears in multiple takes,
  always prefer the highest take number (latest recording).

- NO DUPLICATES: Never include two versions of the same sentence.
  If you already selected "פרויקט חד פעמי" from Take 7,
  do not also select it from Take 3.

{{TAKE_SELECTOR_HINTS}}

RANKED CANDIDATES:
For some sentences, the system identified multiple takes and ranked them.
The best take is listed first. Use this ranking when choosing:

{{CANDIDATES}}

RETURN FORMAT:
{
  "keep_ranges": [
    { "ids": [12,13,14,15,16,17,18,19,20,21,22,23,24], "reason": "HOOK — opening question about needing more employees" },
    { "ids": [34,35,36,37,38,39,40,41,42,43], "reason": "PROBLEM — business built for manual work not growth" },
    ...
  ]
}

CRITICAL: List EVERY SINGLE ID in a chosen segment. Do not skip numbers.
If the segment spans ID 40 to 50, output [40,41,42,43,44,45,46,47,48,49,50].

Numbered text:
`;

interface FlaggedKeepRange extends KeepRange {
  flagged: boolean;
  presenterRatio: number;
}

interface AIFinalReviewResult {
  keep_ranges: KeepRange[];
}

/**
 * AI Step 0.5: Verify duplicate candidates found by take_selector.
 * The math layer found phrases that LOOK similar.
 * The AI verifies they are truly the same content and picks the best version.
 *
 * Returns: updated remove_ids and decisions with AI-verified duplicates.
 */
async function runAIDuplicateVerification(
  duplicateCandidates: DuplicateCandidate[],
  existingRemoveIds: Set<number>,
  brain: AIBrain,
  logger: Logger,
): Promise<{ verifiedRemoveIds: Set<number>; verifiedDecisions: TakeDecision[] }> {

  if (!duplicateCandidates || duplicateCandidates.length === 0) {
    logger.info('AI Duplicate Verification: no candidates to verify');
    return { verifiedRemoveIds: new Set(), verifiedDecisions: [] };
  }

  // Build prompt with all candidate groups
  let groupsText = '';
  for (let i = 0; i < duplicateCandidates.length; i++) {
    const group = duplicateCandidates[i]!;
    groupsText += `\n--- GROUP ${i + 1}: "${group.phrase_text}" (math similarity: ${group.similarity}) ---\n`;

    for (let j = 0; j < group.versions.length; j++) {
      const ver = group.versions[j]!;
      const alreadyRemoved = ver.word_ids.some((id: number) => existingRemoveIds.has(id));
      const status = alreadyRemoved ? ' [ALREADY REMOVED by other rule]' : '';
      groupsText += `  Version ${j + 1} (Take ${ver.take_idx}): "${ver.text}"\n`;
      groupsText += `    Speaker score: ${ver.speaker_avg}, Quality score: ${ver.score}${status}\n`;
      groupsText += `    Word IDs: [${ver.word_ids.join(',')}]\n`;
    }
  }

  const prompt = `You are a Hebrew video editor analyzing raw footage transcripts.
The math layer found ${duplicateCandidates.length} groups of phrases that look similar.
Your job: verify each group and pick the BEST version.

RULES:
1. Two phrases are duplicates ONLY if they express the SAME idea/sentence (even if wording differs slightly)
2. Do NOT mark phrases as duplicates if they are genuinely different content
3. For each confirmed duplicate group, pick the version with:
   - Highest speaker_score (= most clearly the presenter's voice)
   - Most complete sentence (not cut off mid-word)
   - Best flow (natural speech, no stutters)
4. If a version is already marked [ALREADY REMOVED], ignore it — don't pick it as best

CANDIDATE GROUPS:
${groupsText}

Respond ONLY with valid JSON (no markdown, no backticks):
{
  "verified_groups": [
    {
      "group_index": 1,
      "is_duplicate": true,
      "best_version_index": 1,
      "reason": "short explanation in Hebrew"
    }
  ]
}

If a group is NOT a real duplicate, set is_duplicate: false.
Include ALL groups in your response.`;

  try {
    const result = await askAIJSON<{ verified_groups: Array<{
      group_index: number;
      is_duplicate: boolean;
      best_version_index?: number;
      reason?: string;
    }> }>(prompt, {
      brain,
      maxTokens: 2048,
      timeout: 120000,
      logger,
    });

    const data = result.data;
    if (!data || !data.verified_groups) {
      logger.warn('AI Duplicate Verification: invalid response, skipping');
      return { verifiedRemoveIds: new Set(), verifiedDecisions: [] };
    }

    const newRemoveIds = new Set<number>();
    const newDecisions: TakeDecision[] = [];

    for (const vg of data.verified_groups) {
      if (!vg.is_duplicate) continue;

      const groupIdx = vg.group_index - 1; // 1-indexed in prompt
      if (groupIdx < 0 || groupIdx >= duplicateCandidates.length) continue;

      const group = duplicateCandidates[groupIdx]!;
      const bestVerIdx = (vg.best_version_index || 1) - 1;
      if (bestVerIdx < 0 || bestVerIdx >= group.versions.length) continue;

      const bestVersion = group.versions[bestVerIdx]!;
      const bestIds = new Set(bestVersion.word_ids);

      // Remove all other versions
      for (let v = 0; v < group.versions.length; v++) {
        if (v === bestVerIdx) continue;
        const ver = group.versions[v]!;
        const idsToRemove = ver.word_ids.filter(
          (id: number) => !existingRemoveIds.has(id) && !bestIds.has(id)
        );

        if (idsToRemove.length > 0) {
          for (const id of idsToRemove) newRemoveIds.add(id);
          newDecisions.push({
            ids: idsToRemove,
            reason: 'duplicate_take',
            kept_ids: bestVersion.word_ids,
          });
        }
      }

      logger.info(`AI verified duplicate group ${vg.group_index}: ` +
        `keeping version ${vg.best_version_index}, reason: ${vg.reason}`);
    }

    logger.info(`AI Duplicate Verification: ${newRemoveIds.size} additional words to remove`);
    return { verifiedRemoveIds: newRemoveIds, verifiedDecisions: newDecisions };

  } catch (err) {
    logger.error('AI Duplicate Verification failed', { error: String(err) });
    return { verifiedRemoveIds: new Set(), verifiedDecisions: [] };
  }
}

async function runAINarrativeSelection(
  allWordsText: string,
  hints: string,
  candidatesText: string,
  brain: AIBrain,
  logger: Logger,
  retryPreamble?: string,
): Promise<{ narrativeRanges: KeepRange[]; usage: AIUsage }> {
  const prompt = NARRATIVE_PROMPT
    .replace('{{TAKE_SELECTOR_HINTS}}', hints)
    .replace('{{CANDIDATES}}', candidatesText);
  const userPrompt = (retryPreamble ? retryPreamble + '\n\n' : '') + prompt + allWordsText;

  const aiConfig = loadConfig();
  const aiSettings = (aiConfig as unknown as Record<string, unknown>).ai as
    { timeout?: number; maxTokens?: number } | undefined;

  logger.info('AI Step 1: Selecting narrative from all words', { brain });

  let data: AICleanResult;
  let usage: AIUsage;

  try {
    const result = await askAIJSON<AICleanResult>(userPrompt, {
      brain,
      maxTokens: aiSettings?.maxTokens ?? 4096,
      timeout: aiSettings?.timeout ?? 60000,
      logger,
    });
    data = result.data;
    usage = result.usage;
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    logger.warn('AI Step 1 returned non-JSON, retrying with strict JSON instruction', { error: message });

    const jsonRetryPrompt = 'You must return ONLY JSON. No Hebrew text.\n\n' + userPrompt;
    const retryResult = await askAIJSON<AICleanResult>(jsonRetryPrompt, {
      brain,
      maxTokens: aiSettings?.maxTokens ?? 4096,
      timeout: aiSettings?.timeout ?? 60000,
      logger,
    });
    data = retryResult.data;
    usage = retryResult.usage;
  }

  if (!Array.isArray(data.keep_ranges)) {
    data.keep_ranges = [];
  }

  const totalKept = data.keep_ranges.reduce((sum, r) => sum + r.ids.length, 0);
  logger.info('AI Step 1 completed', {
    keepRanges: data.keep_ranges.length,
    totalSelectedWords: totalKept,
  });

  return { narrativeRanges: data.keep_ranges, usage };
}

// ── Cross-reference with presenter detection ──

function crossReferencePresenter(
  ranges: KeepRange[],
  words: Word[],
  logger: Logger,
): FlaggedKeepRange[] {
  const wordsMap = new Map<number, Word>();
  for (const w of words) {
    wordsMap.set(w.id, w);
  }

  const flaggedRanges: FlaggedKeepRange[] = ranges.map((range) => {
    const presenterCount = range.ids.filter((id) => {
      const word = wordsMap.get(id);
      return word?.is_presenter === true;
    }).length;

    const presenterRatio = range.ids.length > 0 ? presenterCount / range.ids.length : 0;
    const flagged = presenterRatio < 0.5;

    if (flagged) {
      logger.info('Flagged segment — likely non-presenter', {
        ids: `${range.ids[0]}-${range.ids[range.ids.length - 1]}`,
        presenterRatio: Math.round(presenterRatio * 100) / 100,
        reason: range.reason,
      });
    }

    return { ...range, flagged, presenterRatio };
  });

  const flaggedCount = flaggedRanges.filter((r) => r.flagged).length;
  logger.info('Cross-reference completed', {
    totalRanges: flaggedRanges.length,
    flaggedRanges: flaggedCount,
  });

  return flaggedRanges;
}

// ── AI Step 2: Final review of flagged segments ──

function buildFinalReviewPrompt(flaggedRanges: FlaggedKeepRange[], allWordsText: string): string {
  const flaggedDescriptions = flaggedRanges
    .filter((r) => r.flagged)
    .map((r) => `- Segment (IDs ${r.ids[0]}-${r.ids[r.ids.length - 1]}): presenterRatio=${Math.round(r.presenterRatio * 100) / 100} — "${r.reason}"`)
    .join('\n');

  return `You selected these segments for the final video. Some segments were flagged because less than 50% of their words match the presenter's voice (they may be the production assistant reading the same script).

FLAGGED SEGMENTS:
${flaggedDescriptions}

ALL SELECTED SEGMENTS:
${JSON.stringify(flaggedRanges.map((r) => ({ ids: r.ids, reason: r.reason, flagged: r.flagged, presenterRatio: Math.round(r.presenterRatio * 100) / 100 })), null, 2)}

For each flagged segment:
1. Is there another take of the same sentence that IS the presenter? → swap it (use IDs from the transcript below)
2. If no presenter version exists → remove the segment entirely

Return your FINAL keep_ranges after corrections.

RETURN FORMAT:
{
  "keep_ranges": [
    { "ids": [50,51,52,53,54,55], "reason": "hook - clearest take" },
    ...
  ]
}

Full transcript for reference:
${allWordsText}`;
}

async function runAIFinalReview(
  flaggedRanges: FlaggedKeepRange[],
  allWordsText: string,
  brain: AIBrain,
  logger: Logger,
): Promise<{ finalRanges: KeepRange[]; usage: AIUsage }> {
  const hasFlagged = flaggedRanges.some((r) => r.flagged);

  if (!hasFlagged) {
    logger.info('AI Step 2: No flagged segments, skipping final review');
    const cleanRanges = flaggedRanges.map(({ ids, reason }) => ({ ids, reason }));
    return { finalRanges: cleanRanges, usage: { inputTokens: 0, outputTokens: 0, estimatedCostUSD: 0 } };
  }

  const userPrompt = buildFinalReviewPrompt(flaggedRanges, allWordsText);

  const aiConfig = loadConfig();
  const aiSettings = (aiConfig as unknown as Record<string, unknown>).ai as
    { timeout?: number; maxTokens?: number } | undefined;

  logger.info('AI Step 2: Final review of flagged segments', {
    brain,
    flaggedCount: flaggedRanges.filter((r) => r.flagged).length,
  });

  const { data, usage } = await askAIJSON<AIFinalReviewResult>(userPrompt, {
    brain,
    maxTokens: aiSettings?.maxTokens ?? 4096,
    timeout: aiSettings?.timeout ?? 60000,
    logger,
  });

  if (!Array.isArray(data.keep_ranges)) {
    data.keep_ranges = [];
  }

  const totalKept = data.keep_ranges.reduce((sum, r) => sum + r.ids.length, 0);
  logger.info('AI Step 2 completed', {
    keepRanges: data.keep_ranges.length,
    totalSelectedWords: totalKept,
  });

  return { finalRanges: data.keep_ranges, usage };
}

// ── Legacy: Semantic cleanup (single AI call) ──

const CLEANUP_PROMPT = `You are a professional video editor building a marketing video from raw footage.

You receive ALL words from the transcript, numbered by ID. The presenter
recorded multiple takes of the same script. Your job: BUILD THE BEST
POSSIBLE VIDEO by selecting the strongest continuous segments.

INSTRUCTIONS:

1. READ the entire transcript and understand the marketing message.
   The typical structure is: Hook → Problem → Solution → Proof → CTA.

2. For each part of the message, IDENTIFY ALL TAKES (versions) that
   exist in the transcript. The presenter often said the same sentence
   3-5 times.

3. For each part, SELECT THE BEST TAKE — the one that is:
   - Most complete (full sentence, not cut off)
   - Most fluent (no stuttering, no hesitation)
   - Last take is usually best (presenter improved over time)

4. Return ONLY the word IDs you want to KEEP, organized as continuous
   segments. Each segment should be a complete thought/sentence.

5. RULES:
   - Never mix words from different takes into one sentence
   - Each segment must be a continuous run of IDs (e.g., [90,91,92,93])
   - Prefer longer, complete takes over short fragments
   - {{STARRED_WORDS_RULE}}
   - Words marked with ~ were flagged as duplicates — usually exclude them
   - The final video should be 30-60 seconds for a service ad
   - CRITICAL: List EVERY SINGLE ID in a chosen take. Do not skip numbers.
     If the take spans ID 40 to 50, output [40,41,42,43,44,45,46,47,48,49,50].
     Missing even one ID will cut a word from the video.
   - INTERNAL RETAKES — CRITICAL:
     Sometimes the presenter says the same sentence TWICE within one continuous
     section (without a long pause). You MUST detect this.

     HOW TO DETECT: If you see the same 3+ word sequence appearing twice
     in your selected IDs (e.g., "פרויקט חד פעמי" appears at IDs 66-68
     AND again at IDs 82-84), you have an internal retake.

     WHAT TO DO: Keep ONLY the LAST (highest ID) version. Start your segment
     from the restart point. Never output a segment that contains duplicate phrases.

     ALSO: If you see stutters like "לא לא" or "מחליף מחליף" (same word twice
     in a row), keep only one occurrence.

   - COMPLETE SENTENCES ONLY — THIS IS CRITICAL:
     BEFORE outputting your final keep_ranges, review EACH segment and verify:

     1. Does it START with the beginning of a sentence?
        BAD: "לך שהחיסכון החודשי..." (starts mid-sentence with "לך")
        BAD: "שלך הוא לא..." (missing "העסק")
        BAD: "חד פעמי..." (missing "זה פרויקט")
        BAD: "לוקחים 23..." (missing "אנחנו")
        BAD: "שהחיסכון..." (starts with ש' prefix — always mid-sentence)
        GOOD: "ואם אני לא מראה לך שהחיסכון החודשי..."
        GOOD: "העסק שלך הוא לא בנוי לצמיחה הוא בנוי לעבודה ידנית"
        GOOD: "אנחנו לוקחים 2-3 תהליכים שגונבים לך הכי הרבה זמן"

     2. Does it END with a complete thought?
        BAD: "...ולא מחליף" (hanging, no object)
        BAD: "...המערכת לא" (hanging "לא")
        BAD: "...מחליף את" (hanging "את")
        GOOD: "...ולא מחליף כלים" or "...ולא מחליף את הצוות"

     3. Can it stand ALONE as a meaningful statement?
        BAD: "מאחורי הקלעים התוצאה" (what result? incomplete)
        GOOD: "מאחורי הקלעים התוצאה היא פשוטה" (complete thought)

     If a segment fails ANY of these checks — either extend it to include
     the missing words, find a better version elsewhere, or REMOVE IT ENTIRELY.

     A 25-second video with 5 perfect segments beats a 35-second video
     with 8 segments where 3 are broken.

   - MINIMUM SEGMENT LENGTH: Each segment you select should be at least
     4 words and approximately 2+ seconds long. If a segment would be
     shorter than this, it's probably a fragment — either find a longer
     version of that sentence elsewhere in the transcript, or skip it entirely.
     A video with 6 strong segments is better than 9 segments with 3 fragments.

   - DO NOT SPLIT SENTENCES: If a sentence like "ואם אני לא מראה לך
     שהחיסכון החודשי הוא גדול מעלות המערכת" is broken by starred words
     in the middle, do NOT output it as two separate segments. Either:
     (a) Find a complete unbroken version of that sentence elsewhere, OR
     (b) Output ONE segment with all the IDs including the gap, OR
     (c) Skip the sentence entirely.
     Two fragments of the same sentence is NEVER acceptable.

RETURN FORMAT:
{
  "keep_ranges": [
    { "ids": [12,13,14,15,16,17,18,19,20], "reason": "hook - best take" },
    { "ids": [90,91,92,93,94,95,96,97], "reason": "solution - final take, most fluent" },
    { "ids": [120,121,122,123,124], "reason": "proof - ROI guarantee" },
    { "ids": [140,141,142,143], "reason": "CTA - clear call to action" }
  ]
}

Numbered text:
`;

function getStarredWordsRule(vadFallback: boolean): string {
  if (vadFallback) {
    return `Words marked with * were flagged by automated voice analysis, but since
     this video has a non-standard camera angle, these flags are UNRELIABLE.
     Treat starred words as REGULAR presenter words — include them freely
     when they form part of a sentence. Only exclude starred words if they
     are clearly production instructions ("סיימתי", "יופי", "מעולה").`;
  }
  return `Words marked with * are non-presenter speech (background voice, crew,
     production assistant). Do NOT include them in your keep_ranges.
     The ONLY exception: if a single starred word (not a sequence of 2+ starred
     words) appears between two presenter words and is clearly the same speaker
     continuing the same sentence, you may include it.
     If you see 2 or more consecutive starred words — they are definitely
     non-presenter. NEVER include a sequence of starred words.`;
}

function readVadFallback(jobDir: string): boolean {
  const segmentsPath = join(jobDir, 'presenter_segments.json');
  if (!existsSync(segmentsPath)) {
    return false;
  }
  try {
    const data = JSON.parse(readFileSync(segmentsPath, 'utf-8')) as { vad_fallback?: boolean };
    return data.vad_fallback === true;
  } catch {
    return false;
  }
}

async function runSemanticCleanup(
  jobDir: string,
  merged: MergedTranscript,
  brain: AIBrain,
  logger: Logger,
  takeSelectorIds?: Set<number>,
): Promise<{ cleanResult: AICleanResult; usage: AIUsage }> {
  const vadFallback = readVadFallback(jobDir);
  if (vadFallback) {
    logger.info('VAD fallback detected — using relaxed starred words rule');
  }
  const prompt = CLEANUP_PROMPT.replace('{{STARRED_WORDS_RULE}}', getStarredWordsRule(vadFallback));
  const numberedText = buildNumberedText(merged.words, takeSelectorIds);
  const userPrompt = prompt + numberedText;

  const aiConfig = loadConfig();
  const aiSettings = (aiConfig as unknown as Record<string, unknown>).ai as
    { timeout?: number; maxTokens?: number } | undefined;

  const presenterWords = merged.words.filter((w) => w.is_presenter);
  logger.info('Starting semantic cleanup', { brain, presenterWordCount: presenterWords.length });

  const { data, usage } = await askAIJSON<AICleanResult>(userPrompt, {
    brain,
    maxTokens: aiSettings?.maxTokens ?? 4096,
    timeout: aiSettings?.timeout ?? 60000,
    logger,
  });

  // Ensure keep_ranges is an array
  if (!Array.isArray(data.keep_ranges)) {
    data.keep_ranges = [];
  }

  const totalKept = data.keep_ranges.reduce((sum, r) => sum + r.ids.length, 0);
  const reasons: Record<string, number> = {};
  for (const r of data.keep_ranges) {
    reasons[r.reason] = (reasons[r.reason] ?? 0) + r.ids.length;
  }

  logger.info('AI selection completed', {
    keepRanges: data.keep_ranges.length,
    totalSelectedWords: totalKept,
    selectionReasons: reasons,
  });

  return { cleanResult: data, usage };
}

// ── Validation: ensure keep_ranges are complete sentences ──

function validateKeepRanges(
  keepRanges: KeepRange[],
  words: Word[],
  logger: Logger,
): KeepRange[] {
  const wordMap = new Map<number, Word>();
  for (const w of words) {
    wordMap.set(w.id, w);
  }

  return keepRanges.filter(range => {
    if (range.ids.length < 6) {
      logger.warn('Removing fragment: less than 6 words', { ids: range.ids });
      return false;
    }

    // ── Check forbidden first words ──
    const FORBIDDEN_STARTS = new Set([
      'זמן', 'וכסף', 'ידנית', 'שלך', 'לך', 'שהחיסכון', 'פרטים', 'בחינם',
      'מחליף', 'והופכים', 'אותם', 'ולא', 'ובלי', 'שגונבים', 'ומחכה', 'ואם',
      'שיהיה', 'של', 'עם', 'גם', 'כי', 'אבל', 'רק', 'עוד',
    ]);

    const firstWordId = range.ids[0];
    const firstWord = wordMap.get(firstWordId!);
    if (firstWord) {
      const firstText = firstWord.word.trim();
      // Check exact match OR starts with ש/ו/ב/ל prefix + 3+ chars (Hebrew prefix pattern)
      const startsWithPrefix = firstText.length >= 4 && ['ש', 'ו', 'ב', 'ל', 'מ', 'כ'].includes(firstText[0]!)
        && !['שלום', 'שאלה', 'לקוח', 'לקוחות', 'מאחורי', 'מערכת', 'מערכות', 'בנוי', 'בינה', 'כלים', 'לומד'].includes(firstText);

      if (FORBIDDEN_STARTS.has(firstText) || startsWithPrefix) {
        logger.warn('Removing segment: starts with forbidden word', {
          word: firstText,
          ids: range.ids,
        });
        return false;
      }
    }

    // ── Check bad ending words ──
    const BAD_ENDS = new Set([
      'לא', 'את', 'של', 'עם', 'גם', 'כי', 'אם', 'או', 'הוא', 'היא',
      'זה', 'אני', 'אתה', 'על', 'בלי', 'עוד', 'רק', 'כל', 'מה',
    ]);

    const lastWordId = range.ids[range.ids.length - 1];
    const lastWord = wordMap.get(lastWordId!);
    if (lastWord && BAD_ENDS.has(lastWord.word.trim())) {
      // Trim the bad ending word instead of removing the whole segment
      const trimmedIds = range.ids.slice(0, -1);
      if (trimmedIds.length >= 6) {
        logger.info('Trimmed bad ending word', {
          word: lastWord.word.trim(),
          originalLength: range.ids.length,
          newLength: trimmedIds.length,
        });
        range.ids = trimmedIds;
        // Check again — maybe the new last word is also bad
        const newLastId = trimmedIds[trimmedIds.length - 1];
        const newLastWord = wordMap.get(newLastId!);
        if (newLastWord && BAD_ENDS.has(newLastWord.word.trim())) {
          range.ids = trimmedIds.slice(0, -1);
          logger.info('Trimmed second bad ending word', {
            word: newLastWord.word.trim(),
          });
        }
        // If after trimming < 6 words, remove entirely
        if (range.ids.length < 6) {
          logger.warn('Removing segment: too short after trimming bad end', { ids: range.ids });
          return false;
        }
      } else {
        logger.warn('Removing segment: bad ending and too short to trim', {
          word: lastWord.word.trim(),
          ids: range.ids,
        });
        return false;
      }
    }

    // Check: does it start mid-sentence?
    if (firstWord) {
      const prevId = range.ids[0]! - 1;
      const prevWord = wordMap.get(prevId);
      if (prevWord) {
        const gap = firstWord.start - prevWord.end;
        if (gap < 0.3) {
          logger.warn('Segment starts close to previous word', {
            firstWord: firstWord.word,
            prevWord: prevWord.word,
            gap,
          });
          // Do NOT trim — the AI chose this range intentionally
        }
      }
    }

    return true;
  });
}

// ── Split ranges that cross take breaks (gap > 1s between consecutive words) ──

function splitRangesAtTakeBreaks(
  keepRanges: KeepRange[],
  words: Word[],
  logger: Logger,
): KeepRange[] {
  const wordsMap = new Map<number, Word>();
  for (const w of words) {
    wordsMap.set(w.id, w);
  }

  const result: KeepRange[] = [];

  for (const range of keepRanges) {
    let currentSegment: number[] = [];

    for (let i = 0; i < range.ids.length; i++) {
      const id = range.ids[i]!;
      currentSegment.push(id);

      // Check if there's a take break after this word
      if (i < range.ids.length - 1) {
        const nextId = range.ids[i + 1]!;
        const currentWord = wordsMap.get(id);
        const nextWord = wordsMap.get(nextId);

        if (currentWord && nextWord) {
          // Split if different takes OR time gap > 1s
          const differentTake = currentWord.take_id !== undefined
            && nextWord.take_id !== undefined
            && currentWord.take_id !== nextWord.take_id;
          const timeGap = nextWord.start - currentWord.end > 1.0;

          if (differentTake || timeGap) {
            logger.warn('Splitting range at take break', {
              differentTake,
              timeGap: nextWord.start - currentWord.end,
              takeCurrent: currentWord.take_id,
              takeNext: nextWord.take_id,
              beforeWord: currentWord.word,
              afterWord: nextWord.word,
              segmentLength: currentSegment.length,
            });
            if (currentSegment.length >= 4) {
              result.push({
                ids: [...currentSegment],
                reason: range.reason + ' [split at take break]',
              });
            }
            currentSegment = [];
          }
        }
      }
    }

    // Add remaining segment
    if (currentSegment.length >= 4) {
      result.push({
        ids: currentSegment,
        reason: range.reason,
      });
    }
  }

  logger.info('splitRangesAtTakeBreaks', {
    before: keepRanges.length,
    after: result.length,
  });

  return result;
}

// ── Enforce starred words limit (max consecutive non-presenter words per segment) ──

function enforceStarredLimit(
  keepRanges: KeepRange[],
  words: Word[],
  logger: Logger,
  maxConsecutiveStarred: number = 1,
): KeepRange[] {
  const wordsMap = new Map<number, Word>();
  for (const w of words) {
    wordsMap.set(w.id, w);
  }

  const result: KeepRange[] = [];

  for (const range of keepRanges) {
    // Count max consecutive starred (non-presenter) words
    let maxConsec = 0;
    let currentConsec = 0;
    let totalStarred = 0;

    for (const id of range.ids) {
      const word = wordsMap.get(id);
      if (word && !word.is_presenter) {
        currentConsec++;
        totalStarred++;
        maxConsec = Math.max(maxConsec, currentConsec);
      } else {
        currentConsec = 0;
      }
    }

    if (maxConsec > maxConsecutiveStarred) {
      // Try to trim starred words from edges
      const trimmedIds = trimStarredEdges(range.ids, wordsMap);

      if (trimmedIds.length >= 4) {
        // Re-check after trimming
        let stillBad = false;
        let consec = 0;
        for (const id of trimmedIds) {
          const w = wordsMap.get(id);
          if (w && !w.is_presenter) { consec++; } else { consec = 0; }
          if (consec > maxConsecutiveStarred) { stillBad = true; break; }
        }

        if (!stillBad) {
          result.push({ ids: trimmedIds, reason: range.reason + ' [trimmed starred]' });
        } else {
          logger.warn('Removing segment with too many starred words', {
            ids: range.ids, totalStarred, maxConsec,
          });
        }
      } else {
        logger.warn('Removing segment with too many starred words (too short after trim)', {
          ids: range.ids, totalStarred, maxConsec,
        });
      }
    } else {
      result.push(range);
    }
  }

  logger.info('enforceStarredLimit', {
    before: keepRanges.length,
    after: result.length,
    removed: keepRanges.length - result.length,
  });

  return result;
}

function trimStarredEdges(ids: number[], wordsMap: Map<number, Word>): number[] {
  let start = 0;
  let end = ids.length - 1;

  // Trim starred from start
  while (start < ids.length) {
    const w = wordsMap.get(ids[start]!);
    if (w && !w.is_presenter) { start++; } else { break; }
  }

  // Trim starred from end
  while (end >= 0) {
    const w = wordsMap.get(ids[end]!);
    if (w && !w.is_presenter) { end--; } else { break; }
  }

  return ids.slice(start, end + 1);
}

// ── Deduplicate repeated phrases within a single segment ──

function deduplicateWithinSegments(
  keepRanges: KeepRange[],
  words: Word[],
  logger: Logger,
): KeepRange[] {
  const wordsMap = new Map<number, Word>();
  for (const w of words) {
    wordsMap.set(w.id, w);
  }

  const result: KeepRange[] = [];

  for (const range of keepRanges) {
    const rangeWords = range.ids
      .map(id => wordsMap.get(id))
      .filter((w): w is Word => w !== undefined);

    const wordsList = rangeWords.map(w => w.word);
    let foundDuplicate = false;

    // Check if any 4+ word sequence repeats within the segment
    for (let len = Math.floor(wordsList.length / 2); len >= 4; len--) {
      for (let i = 0; i <= wordsList.length - len * 2; i++) {
        const phrase = wordsList.slice(i, i + len).join(' ');
        const rest = wordsList.slice(i + len).join(' ');

        if (rest.includes(phrase)) {
          // Found duplicate — keep the second (later) occurrence
          const midpoint = Math.floor(range.ids.length / 2);
          const secondHalf = range.ids.slice(midpoint);

          if (secondHalf.length >= 4) {
            logger.warn('Deduplicating repeated phrase within segment', {
              phrase,
              originalLength: range.ids.length,
              keptLength: secondHalf.length,
            });
            result.push({
              ids: secondHalf,
              reason: range.reason + ' [deduped - kept later version]',
            });
          } else {
            logger.warn('Removing segment with repeated phrase (too short after dedup)', {
              phrase,
              originalLength: range.ids.length,
            });
          }
          foundDuplicate = true;
          break;
        }
      }
      if (foundDuplicate) break;
    }

    if (!foundDuplicate) {
      result.push(range);
    }
  }

  logger.info('deduplicateWithinSegments', {
    before: keepRanges.length,
    after: result.length,
    deduped: keepRanges.length - result.length,
  });

  return result;
}

// ── Trim duplicate edge words (e.g. "ידנית" from previous sentence leaking into next segment) ──

function trimDuplicateEdges(
  keepRanges: KeepRange[],
  words: Word[],
  logger: Logger,
): KeepRange[] {
  const wordsMap = new Map<number, Word>();
  for (const w of words) {
    wordsMap.set(w.id, w);
  }

  const result: KeepRange[] = [];

  for (const range of keepRanges) {
    const rangeWords = range.ids
      .map(id => wordsMap.get(id))
      .filter((w): w is Word => w !== undefined);

    if (rangeWords.length < 5) {
      result.push(range);
      continue;
    }

    // Check: does the first word appear again later in the segment?
    const firstWordText = rangeWords[0]!.word;
    const laterIndex = rangeWords.findIndex((w, i) => i > 2 && w.word === firstWordText);

    if (laterIndex > 0 && laterIndex < rangeWords.length - 3) {
      // The first word is a stray from the previous sentence — trim it
      const newIds = range.ids.slice(1);
      if (newIds.length >= 4) {
        logger.info('trimDuplicateEdges: removed stray first word', {
          removedWord: firstWordText,
          originalLength: range.ids.length,
          newLength: newIds.length,
        });
        result.push({ ids: newIds, reason: range.reason + ' [trimmed duplicate edge]' });
        continue;
      }
    }

    result.push(range);
  }

  logger.info('trimDuplicateEdges', {
    before: keepRanges.length,
    after: result.length,
  });

  return result;
}

// ── Step 3: Build keep/remove segments from words ──

function buildSegments(
  allWords: Word[],
  removeRanges: KeepRange[],
): { keepSegments: KeepSegment[]; removeSegments: RemoveSegment[] } {
  // Collect all IDs to remove
  const removeIdSet = new Set<number>();
  const removeReasonMap = new Map<number, string>();
  for (const range of removeRanges) {
    for (const id of range.ids) {
      removeIdSet.add(id);
      removeReasonMap.set(id, range.reason);
    }
  }

  // Filter: keep only presenter words that AI didn't mark for removal
  const keptWords: Word[] = [];
  const removedWords: Array<Word & { reason: string }> = [];

  for (const w of allWords) {
    if (removeIdSet.has(w.id)) {
      removedWords.push({ ...w, reason: removeReasonMap.get(w.id) ?? 'ai_removal' });
    } else {
      keptWords.push(w);
    }
  }

  // Sort kept words chronologically to prevent start > end in segments
  keptWords.sort((a, b) => a.start - b.start);

  // Build keep_segments: gap > 0.5s = new segment
  const keepSegments: KeepSegment[] = [];
  if (keptWords.length > 0) {
    let segWords: Word[] = [keptWords[0]!];

    for (let i = 1; i < keptWords.length; i++) {
      const prev = segWords[segWords.length - 1]!;
      const curr = keptWords[i]!;

      if (curr.start - prev.end > 0.5) {
        // Close current segment
        keepSegments.push({
          start: segWords[0]!.start,
          end: segWords[segWords.length - 1]!.end,
          word_ids: segWords.map((w) => w.id),
        });
        segWords = [curr];
      } else {
        segWords.push(curr);
      }
    }

    // Flush last segment
    if (segWords.length > 0) {
      keepSegments.push({
        start: segWords[0]!.start,
        end: segWords[segWords.length - 1]!.end,
        word_ids: segWords.map((w) => w.id),
      });
    }
  }

  // Build remove_segments: group consecutive removed words by reason
  const sortedRemoved = [...removedWords].sort((a, b) => a.start - b.start);
  const removeSegments: RemoveSegment[] = [];

  if (sortedRemoved.length > 0) {
    let segStart = sortedRemoved[0]!.start;
    let segEnd = sortedRemoved[0]!.end;
    let segReason = sortedRemoved[0]!.reason;

    for (let i = 1; i < sortedRemoved.length; i++) {
      const curr = sortedRemoved[i]!;
      // Same reason and close together → extend segment
      if (curr.reason === segReason && curr.start - segEnd <= 0.5) {
        segEnd = curr.end;
      } else {
        removeSegments.push({ start: segStart, end: segEnd, reason: segReason });
        segStart = curr.start;
        segEnd = curr.end;
        segReason = curr.reason;
      }
    }
    removeSegments.push({ start: segStart, end: segEnd, reason: segReason });
  }

  return { keepSegments, removeSegments };
}

/**
 * Step 2.85: Find and remove repeated phrases BETWEEN consecutive segments.
 * If the same 3+ word phrase appears at the end of one segment and
 * the start of the next — trim it from one of them.
 */
function deduplicateBetweenSegments(
  ranges: KeepRange[],
  words: Word[],
  logger?: Logger,
): KeepRange[] {
  if (ranges.length < 2) return ranges;

  const wordMap = new Map(words.map(w => [w.id, w]));

  const getWordsText = (ids: number[]): string[] =>
    ids.map(id => wordMap.get(id)?.word?.trim() || '').filter(Boolean);

  let totalTrimmed = 0;

  for (let i = 0; i < ranges.length - 1; i++) {
    const currentWords = getWordsText(ranges[i]!.ids);
    const nextWords = getWordsText(ranges[i + 1]!.ids);

    if (currentWords.length < 3 || nextWords.length < 3) continue;

    // Check if last N words of current match first N words of next (exact)
    for (let matchLen = Math.min(5, currentWords.length, nextWords.length); matchLen >= 2; matchLen--) {
      const currentTail = currentWords.slice(-matchLen).join(' ');
      const nextHead = nextWords.slice(0, matchLen).join(' ');

      if (currentTail === nextHead) {
        // Remove the duplicate from the END of current segment
        ranges[i]!.ids = ranges[i]!.ids.slice(0, -matchLen);
        logger?.info('Removed cross-segment exact duplicate', {
          phrase: currentTail,
          fromSegment: i + 1,
          wordsRemoved: matchLen,
        });
        totalTrimmed += matchLen;
        break;
      }
    }

    // Check for shared 3-gram anywhere in consecutive segments
    if (currentWords.length >= 3 && nextWords.length >= 3) {
      const currentTrigrams = new Set<string>();
      for (let k = 0; k < currentWords.length - 2; k++) {
        currentTrigrams.add(currentWords.slice(k, k + 3).join(' '));
      }

      for (let k = 0; k < nextWords.length - 2; k++) {
        const nextTrigram = nextWords.slice(k, k + 3).join(' ');
        if (currentTrigrams.has(nextTrigram)) {
          // Found shared trigram — trim it from the START of next segment
          const trimStart = k;
          const trimEnd = k + 3;

          // Only trim if it's at the beginning of the next segment (within first 4 words)
          if (trimStart <= 1) {
            ranges[i + 1]!.ids = ranges[i + 1]!.ids.slice(trimEnd);
            logger?.info('Removed cross-segment trigram duplicate', {
              phrase: nextTrigram,
              fromSegment: i + 2,
              wordsRemoved: 3,
            });
            totalTrimmed += 3;
          }
          break; // Only handle first match per pair
        }
      }
    }
  }

  // Remove segments that became too short after trimming
  const filtered = ranges.filter(r => {
    if (r.ids.length < 6) {
      logger?.warn('Removing segment: too short after cross-segment dedup', { ids: r.ids });
      return false;
    }
    return true;
  });

  if (totalTrimmed > 0) {
    logger?.info('deduplicateBetweenSegments', {
      before: ranges.length,
      after: filtered.length,
      totalWordsTrimmed: totalTrimmed,
    });
  }

  return filtered;
}

// ── Main exported function ──────────────────────

export async function runMergeAndClean(
  jobDir: string,
  brain: AIBrain,
  logger: Logger,
): Promise<void> {
  const statusPath = join(jobDir, 'status.json');
  const startTime = Date.now();

  // Update status: merge-and-clean starting
  const currentStatus = readStatusFile(statusPath);
  const currentProgress = (currentStatus.progress as Record<string, unknown>) ?? {};
  writeStatus(statusPath, {
    currentStep: 'mergeAndClean',
    progress: {
      ...currentProgress,
      mergeAndClean: { status: 'processing' },
    },
  });

  // Step 1: Merge
  let merged = await runMerge(jobDir, logger);

  // Step 1.25: WhisperX alignment (improve word timestamps)
  merged = await runAlignWords(jobDir, logger);

  // Step 1.5: Take selector (rule-based duplicate detection — metadata/hints)
  const takeDecisions = await runTakeSelector(jobDir, logger);
  const takeSelectorIds = new Set(takeDecisions.remove_ids);

  // Build hard-rejected IDs set (words AI should never see)
  const hardRejectedIds = new Set<number>();
  for (const dec of takeDecisions.decisions) {
    if (dec.reason === 'hard_reject') {
      for (const id of dec.ids) {
        hardRejectedIds.add(id);
      }
    }
  }
  logger.info('Hard rejected words (hidden from AI)', { count: hardRejectedIds.size });

  // ── AI Step 0.5: Verify duplicate candidates ──
  const duplicateCandidates = takeDecisions.duplicate_candidates || [];
  if (duplicateCandidates.length > 0) {
    logger.info('AI Step 0.5: Verifying duplicate candidates', {
      groups: duplicateCandidates.length, brain,
    });

    const { verifiedRemoveIds, verifiedDecisions } = await runAIDuplicateVerification(
      duplicateCandidates,
      new Set([...takeSelectorIds, ...hardRejectedIds]),
      brain,
      logger,
    );

    // Merge verified duplicates into take selector results
    for (const id of verifiedRemoveIds) {
      takeSelectorIds.add(id);
    }
    // Add verified decisions to hints
    for (const dec of verifiedDecisions) {
      takeDecisions.decisions.push(dec);
      takeDecisions.remove_ids.push(...dec.ids);
    }

    logger.info('AI Step 0.5 completed', {
      additionalRemovals: verifiedRemoveIds.size,
      totalTakeSelectorIds: takeSelectorIds.size,
    });
  }

  // Step 2: AI Step 1 — Select narrative from ALL words (with take selector hints)
  const allWordsText = buildStructuredText(merged.words, takeSelectorIds, hardRejectedIds, logger);
  const hints = buildTakeSelectorHints(takeDecisions);
  const candidatesText = buildCandidatesText(takeDecisions);
  logger.info('Take selector hints for AI', {
    hintCount: takeDecisions.decisions.length,
    candidateGroups: takeDecisions.candidates?.length ?? 0,
  });
  let { narrativeRanges, usage: usage1 } = await runAINarrativeSelection(allWordsText, hints, candidatesText, brain, logger);

  // Step 2.4: Coverage retry — if below 55%, re-run AI with stronger instruction
  const presenterWords = merged.words.filter(w => w.is_presenter && !hardRejectedIds.has(w.id));
  const initialSelectedIds = new Set(narrativeRanges.flatMap(r => r.ids));
  const initialSelectedPresenter = presenterWords.filter(w => initialSelectedIds.has(w.id)).length;
  const initialCoverage = presenterWords.length > 0 ? initialSelectedPresenter / presenterWords.length : 1;

  if (initialCoverage < 0.55 && presenterWords.length > 0) {
    logger.warn('Coverage too low after AI Step 1, re-running with stronger instruction', {
      coverage: `${(initialCoverage * 100).toFixed(0)}%`,
      selected: initialSelectedPresenter,
      presenter: presenterWords.length,
    });

    const retryPreamble = `Your previous selection covered only ${Math.round(initialCoverage * 100)}% of presenter words.
This is below the 55% minimum. You MUST add more segments.

MISSING PARTS CHECK — verify each one:
- Hook/opening question
- Problem statement
- Solution description
- How it works (process)
- Result/guarantee
- Call to action

Add segments for any missing part. Target: at least ${Math.ceil(presenterWords.length * 0.6)} words.

RETURN FORMAT — JSON ONLY:
Respond with ONLY a valid JSON object. No text before or after.
{
  "keep_ranges": [
    { "ids": [1,2,3,...], "reason": "..." },
    ...
  ]
}`;

    const retry = await runAINarrativeSelection(allWordsText, hints, candidatesText, brain, logger, retryPreamble);
    narrativeRanges = retry.narrativeRanges;
    usage1 = {
      inputTokens: usage1.inputTokens + retry.usage.inputTokens,
      outputTokens: usage1.outputTokens + retry.usage.outputTokens,
      estimatedCostUSD: usage1.estimatedCostUSD + retry.usage.estimatedCostUSD,
    };

    const retrySelectedIds = new Set(narrativeRanges.flatMap(r => r.ids));
    const retryCoverage = presenterWords.filter(w => retrySelectedIds.has(w.id)).length / presenterWords.length;
    logger.info('Coverage after retry', { coverage: `${(retryCoverage * 100).toFixed(0)}%` });
  }

  // Step 2.5: Validate keep_ranges — remove fragments and log suspicious starts
  const validatedRanges = validateKeepRanges(narrativeRanges, merged.words, logger);
  logger.info('Validation completed', {
    before: narrativeRanges.length,
    after: validatedRanges.length,
    removed: narrativeRanges.length - validatedRanges.length,
  });

  // Step 2.6: Split ranges that cross take breaks (gap > 1s)
  const splitRanges = splitRangesAtTakeBreaks(validatedRanges, merged.words, logger);

  // Step 2.7: Enforce starred words limit (max 1 consecutive non-presenter)
  const starredEnforced = enforceStarredLimit(splitRanges, merged.words, logger);

  // Step 2.8: Deduplicate repeated phrases within segments
  const dedupedRanges = deduplicateWithinSegments(starredEnforced, merged.words, logger);

  // Step 2.85: Remove repeated phrases between consecutive segments
  const crossDedupedRanges = deduplicateBetweenSegments(dedupedRanges, merged.words, logger);

  // Step 2.9: Trim duplicate edge words (e.g. stray word from previous sentence)
  const trimmedRanges = trimDuplicateEdges(crossDedupedRanges, merged.words, logger);

  // Step 3: Cross-reference with presenter detection
  const flaggedRanges = crossReferencePresenter(trimmedRanges, merged.words, logger);

  // Step 4: AI Step 2 — Final review (swap/remove flagged segments)
  const { finalRanges, usage: usage2 } = await runAIFinalReview(flaggedRanges, allWordsText, brain, logger);

  // Combine usage from both AI calls
  const usage: AIUsage = {
    inputTokens: usage1.inputTokens + usage2.inputTokens,
    outputTokens: usage1.outputTokens + usage2.outputTokens,
    estimatedCostUSD: usage1.estimatedCostUSD + usage2.estimatedCostUSD,
  };

  // Build effective remove ranges from AI keep_ranges (invert: everything not kept is removed)
  const keepIdSet = new Set(finalRanges.flatMap(r => r.ids));
  const effectiveRemoveRanges: KeepRange[] = [];
  if (keepIdSet.size > 0) {
    const notKeptIds = merged.words
      .filter(w => !keepIdSet.has(w.id))
      .map(w => w.id);
    if (notKeptIds.length > 0) {
      effectiveRemoveRanges.push({ ids: notKeptIds, reason: 'not_selected_by_ai' });
    }
  }

  // Step 5: Build keep/remove segments from word-level data
  const { keepSegments: rawKeepSegments, removeSegments } = buildSegments(merged.words, effectiveRemoveRanges);

  // Filter out invalid segments (start >= end) and micro-segments (< 0.8s)
  const MIN_KEEP_SEGMENT_DURATION = 2.0;
  const keepSegments = rawKeepSegments.filter((seg) => {
    if (seg.start >= seg.end) {
      logger.warn('Skipping invalid segment: start >= end', { start: seg.start, end: seg.end, duration: Math.round((seg.end - seg.start) * 1000) / 1000 });
      return false;
    }
    const duration = seg.end - seg.start;
    if (duration < MIN_KEEP_SEGMENT_DURATION) {
      logger.info('Removing micro-segment', { start: seg.start, end: seg.end, duration: Math.round(duration * 1000) / 1000 });
      return false;
    }
    return true;
  });

  const processingTimeMs = Date.now() - startTime;

  // Count stats
  const selectedByAI = keepIdSet.size;
  const keptWords = merged.words.filter((w) => keepIdSet.has(w.id));
  const flaggedSegmentCount = flaggedRanges.filter((r) => r.flagged).length;
  const aiCalls = usage2.inputTokens > 0 ? 2 : 1;

  // Coverage validation
  const presenterWordCount = merged.words.filter((w) => w.is_presenter).length;
  if (presenterWordCount > 0) {
    const coverage = selectedByAI / presenterWordCount;
    if (coverage < 0.50) {
      logger.warn('AI coverage too low, consider re-running', {
        selected: selectedByAI,
        presenter: presenterWordCount,
        coverage: `${(coverage * 100).toFixed(0)}%`,
        target: '55-80%',
      });
    } else {
      logger.info('AI coverage', {
        selected: selectedByAI,
        presenter: presenterWordCount,
        coverage: `${(coverage * 100).toFixed(0)}%`,
      });
    }
  }

  // Build final output
  const cleanedTranscript = {
    words: merged.words,
    keep_segments: keepSegments,
    remove_segments: removeSegments,
    stats: {
      original_words: merged.stats.total_words,
      presenter_words: merged.stats.presenter_words,
      cleaned_words: keptWords.length,
      removed_by_merge: merged.stats.other_words,
      selected_by_ai: selectedByAI,
      take_selector_hints: takeSelectorIds.size,
      flagged_segments: flaggedSegmentCount,
      ai_calls: aiCalls,
      ai_brain: brain,
      ai_cost_usd: usage.estimatedCostUSD,
      processing_time_ms: processingTimeMs,
    },
  };

  // Save cleaned_transcript.json
  const outputPath = join(jobDir, 'cleaned_transcript.json');
  writeFileSync(outputPath, JSON.stringify(cleanedTranscript, null, 2), 'utf-8');

  logger.info('Merge and clean completed', {
    outputPath,
    originalWords: merged.stats.total_words,
    presenterWords: merged.stats.presenter_words,
    cleanedWords: keptWords.length,
    removedByMerge: merged.stats.other_words,
    selectedByAI: selectedByAI,
    takeSelectorHints: takeSelectorIds.size,
    flaggedSegments: flaggedSegmentCount,
    aiCalls,
    keepSegments: keepSegments.length,
    removeSegments: removeSegments.length,
    brain,
    costUSD: usage.estimatedCostUSD.toFixed(4),
    processingTimeMs,
  });

  // Update status
  const finalStatus = readStatusFile(statusPath);
  const finalProgress = (finalStatus.progress as Record<string, unknown>) ?? {};

  // Update AI costs in status
  const existingCosts = (finalStatus.aiCosts as { totalCalls?: number; totalInputTokens?: number; totalOutputTokens?: number; estimatedCostUSD?: number }) ?? {};

  writeStatus(statusPath, {
    status: 'cleaned',
    currentStep: 'done',
    progress: {
      ...finalProgress,
      mergeAndClean: { status: 'done', durationMs: processingTimeMs },
    },
    mergeAndClean: {
      originalWords: merged.stats.total_words,
      presenterWords: merged.stats.presenter_words,
      cleanedWords: keptWords.length,
      removedByMerge: merged.stats.other_words,
      selectedByAI: selectedByAI,
      takeSelectorHints: takeSelectorIds.size,
      flaggedSegments: flaggedSegmentCount,
      aiCalls,
      keepSegments: keepSegments.length,
      brain,
      outputPath,
    },
    aiCosts: {
      totalCalls: (existingCosts.totalCalls ?? 0) + aiCalls,
      totalInputTokens: (existingCosts.totalInputTokens ?? 0) + usage.inputTokens,
      totalOutputTokens: (existingCosts.totalOutputTokens ?? 0) + usage.outputTokens,
      estimatedCostUSD: Number(((existingCosts.estimatedCostUSD ?? 0) + usage.estimatedCostUSD).toFixed(4)),
    },
  });
}

// ── Pipeline step wrapper ────────────────────────

async function mergeAndClean(context: StepContext): Promise<StepResult> {
  const { outputDir, logger } = context;

  // Read settings to determine which AI brain to use
  const settingsPath = join(outputDir, 'settings.json');
  let brain: AIBrain = 'claude_sonnet_4_6';
  if (existsSync(settingsPath)) {
    try {
      const settings = JSON.parse(readFileSync(settingsPath, 'utf-8')) as { aiBrain?: string };
      if (settings.aiBrain === 'gpt_5_4') {
        brain = 'gpt_5_4';
      }
    } catch {
      // Use default
    }
  }

  await runMergeAndClean(outputDir, brain, logger);

  return {
    outputFile: context.originalFile,
    success: true,
    message: 'Merge and semantic cleanup completed',
  };
}

registerStep('merge-and-clean', mergeAndClean);
