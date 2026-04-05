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

interface TakeDecisions {
  remove_ids: number[];
  decisions: TakeDecision[];
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

interface Sentence {
  id: number;
  text: string;
  word_ids: number[];
  take_id: number;
  start: number;
  end: number;
  duration: number;
  score: number;
  word_count: number;
  speaker_score_avg: number;
  completeness: number;
}

interface SentenceBuilderOutput {
  sentences: Sentence[];
  stats: {
    total_takes: number;
    total_raw_sentences: number;
    after_dedup: number;
    total_duration: number;
    avg_score: number;
    processing_time_ms: number;
  };
}

interface AISentenceSelectionResult {
  selected_ids: number[];
  structure?: Record<string, number[]>;
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

// ── Step 2: Sentence-based cleanup ──────────────

/** Simple numbered text for AI final review reference */
function buildNumberedTextRaw(words: Word[]): string {
  return words.map((w) => `[${w.id}] ${w.word}`).join(' ');
}

// ── Step 2a: Run sentence_builder.py ──

async function runSentenceBuilder(
  jobDir: string,
  logger: Logger,
): Promise<SentenceBuilderOutput> {
  const mergedPath = join(jobDir, 'merged_transcript.json');
  const takeDecisionsPath = join(jobDir, 'take_decisions.json');
  const outputPath = join(jobDir, 'sentences.json');

  const scriptPath = resolve('python', 'sentence_builder.py');

  const config = loadConfig();
  const pdConfig = (config as unknown as Record<string, unknown>).presenterDetection as
    { pythonPath?: string } | undefined;
  const pythonPath = pdConfig?.pythonPath ?? 'python3';

  const args = [
    scriptPath,
    '--merged', mergedPath,
    '--take-decisions', takeDecisionsPath,
    '--output', outputPath,
  ];

  logger.info('Running sentence_builder.py', { mergedPath, takeDecisionsPath });

  const { stdout, stderr } = await execFileAsync(pythonPath, args, {
    timeout: 120000,
  });

  if (stderr) {
    logger.debug('sentence_builder.py stderr', { stderr: stderr.trim() });
  }

  const result = JSON.parse(stdout) as SentenceBuilderOutput;

  logger.info('Sentence builder completed', {
    totalSentences: result.sentences.length,
    totalDuration: result.stats.total_duration,
    avgScore: result.stats.avg_score,
    rawSentences: result.stats.total_raw_sentences,
    afterDedup: result.stats.after_dedup,
  });

  return result;
}

// ── Step 2b: Build sentence menu for AI ──

function buildSentenceMenu(sentences: Sentence[]): string {
  return sentences.map(s =>
    `[${s.id}] (${s.duration.toFixed(1)}s, score:${s.score.toFixed(2)}) "${s.text}"`
  ).join('\n');
}

const SENTENCE_MENU_PROMPT = `You are the world's best direct-response copywriter building a marketing video.

Below is a MENU of complete sentences extracted from the raw footage.
Each sentence is complete and ready to use — you just need to SELECT
which ones to include and in what ORDER.

MENU:
{{SENTENCE_MENU}}

YOUR TASK:
Select sentences that build the strongest possible marketing video.
Structure: Hook → Problem → Solution → How it works → Proof/Guarantee → CTA

RULES:
1. Select by sentence ID only — return a list of IDs
2. Keep chronological order — if sentence 5 comes before sentence 12
   in the timeline, put 5 before 12
3. Cover the FULL message — include hook, problem, solution, proof, and CTA
4. Select at least 60% of available sentences
5. Never select two sentences that say the same thing

RETURN FORMAT — JSON ONLY:
{
  "selected_ids": [1, 2, 5, 7, 9, 12, 15, 18],
  "structure": {
    "hook": [1],
    "problem": [2],
    "solution": [5, 7],
    "mechanism": [9],
    "proof": [12, 15],
    "cta": [18]
  }
}
`;

// ── Step 2c: AI selects from sentence menu ──

async function runAISentenceSelection(
  menu: string,
  brain: AIBrain,
  logger: Logger,
): Promise<{ selectedIds: number[]; usage: AIUsage }> {
  const userPrompt = SENTENCE_MENU_PROMPT.replace('{{SENTENCE_MENU}}', menu);

  const aiConfig = loadConfig();
  const aiSettings = (aiConfig as unknown as Record<string, unknown>).ai as
    { timeout?: number; maxTokens?: number } | undefined;

  logger.info('AI sentence selection: sending menu to AI', { brain });

  let data: AISentenceSelectionResult;
  let usage: AIUsage;

  try {
    const result = await askAIJSON<AISentenceSelectionResult>(userPrompt, {
      brain,
      maxTokens: aiSettings?.maxTokens ?? 4096,
      timeout: aiSettings?.timeout ?? 60000,
      logger,
    });
    data = result.data;
    usage = result.usage;
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    logger.warn('AI sentence selection returned non-JSON, retrying', { error: message });

    const retryPrompt = 'You must return ONLY JSON. No Hebrew text.\n\n' + userPrompt;
    const retryResult = await askAIJSON<AISentenceSelectionResult>(retryPrompt, {
      brain,
      maxTokens: aiSettings?.maxTokens ?? 4096,
      timeout: aiSettings?.timeout ?? 60000,
      logger,
    });
    data = retryResult.data;
    usage = retryResult.usage;
  }

  if (!Array.isArray(data.selected_ids)) {
    data.selected_ids = [];
  }

  logger.info('AI sentence selection completed', {
    selectedSentences: data.selected_ids.length,
    structure: data.structure ? Object.keys(data.structure) : [],
  });

  return { selectedIds: data.selected_ids, usage };
}

// ── Step 2d: Map selected sentences to keep segments ──

function mapSentencesToSegments(
  selectedIds: number[],
  sentences: Sentence[],
  logger: Logger,
): KeepRange[] {
  const sentMap = new Map(sentences.map(s => [s.id, s]));

  const ranges = selectedIds
    .map(id => sentMap.get(id))
    .filter((s): s is Sentence => {
      if (!s) {
        logger.warn('AI selected unknown sentence ID, skipping');
        return false;
      }
      return true;
    })
    .sort((a, b) => a.start - b.start)
    .map(sent => ({
      ids: sent.word_ids,
      reason: `sentence_${sent.id}`,
    }));

  logger.info('mapSentencesToSegments', {
    selectedIds: selectedIds.length,
    mappedRanges: ranges.length,
  });

  return ranges;
}

// ── Cross-reference with presenter detection ──

interface FlaggedKeepRange extends KeepRange {
  flagged: boolean;
  presenterRatio: number;
}

interface AIFinalReviewResult {
  keep_ranges: KeepRange[];
}

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

  // Step 1.25: WhisperX alignment
  merged = await runAlignWords(jobDir, logger);

  // Step 1.5: Take selector (rule-based)
  const takeDecisions = await runTakeSelector(jobDir, logger);

  // Step 2: Build sentence menu (Python)
  const sentenceData = await runSentenceBuilder(jobDir, logger);
  const sentences = sentenceData.sentences;

  if (sentences.length === 0) {
    logger.warn('No sentences built — cannot proceed with AI selection');
    const emptyOutput = {
      words: merged.words,
      keep_segments: [] as KeepSegment[],
      remove_segments: [] as RemoveSegment[],
      stats: {
        original_words: merged.stats.total_words,
        presenter_words: merged.stats.presenter_words,
        cleaned_words: 0,
        removed_by_merge: merged.stats.other_words,
        selected_by_ai: 0,
        take_selector_hints: takeDecisions.remove_ids.length,
        flagged_segments: 0,
        ai_calls: 0,
        ai_brain: brain,
        ai_cost_usd: 0,
        processing_time_ms: Date.now() - startTime,
      },
    };
    writeFileSync(join(jobDir, 'cleaned_transcript.json'), JSON.stringify(emptyOutput, null, 2), 'utf-8');
    writeStatus(statusPath, {
      status: 'cleaned',
      currentStep: 'done',
      progress: { ...currentProgress, mergeAndClean: { status: 'done', durationMs: Date.now() - startTime } },
    });
    return;
  }

  // Step 3: AI selects from sentence menu
  const menu = buildSentenceMenu(sentences);
  logger.info('Sentence menu built', {
    sentenceCount: sentences.length,
    totalDuration: sentenceData.stats.total_duration,
    menuLength: menu.length,
  });

  let { selectedIds, usage: usage1 } = await runAISentenceSelection(menu, brain, logger);

  // Coverage check — if AI selected < 60%, use all sentences as fallback
  const minRequired = Math.ceil(sentences.length * 0.6);
  if (selectedIds.length < minRequired && sentences.length > 0) {
    logger.warn('AI selected too few sentences, using all as fallback', {
      selected: selectedIds.length,
      minimum: minRequired,
      total: sentences.length,
    });
    selectedIds = sentences.map(s => s.id);
  }

  // Step 4: Map sentences to keep ranges (code, not AI)
  const narrativeRanges = mapSentencesToSegments(selectedIds, sentences, logger);

  // Step 5: Cross-reference with presenter detection (safety check)
  const flaggedRanges = crossReferencePresenter(narrativeRanges, merged.words, logger);

  // Step 6: AI final review (only if flagged segments exist)
  const allWordsText = buildNumberedTextRaw(merged.words);
  const { finalRanges, usage: usage2 } = await runAIFinalReview(flaggedRanges, allWordsText, brain, logger);

  // Combine AI usage
  const usage: AIUsage = {
    inputTokens: usage1.inputTokens + usage2.inputTokens,
    outputTokens: usage1.outputTokens + usage2.outputTokens,
    estimatedCostUSD: usage1.estimatedCostUSD + usage2.estimatedCostUSD,
  };

  // Build effective remove ranges (invert: everything not kept is removed)
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

  // Step 7: Build keep/remove segments
  const { keepSegments: rawKeepSegments, removeSegments } = buildSegments(merged.words, effectiveRemoveRanges);

  // Filter out invalid segments
  const MIN_KEEP_SEGMENT_DURATION = 2.0;
  const keepSegments = rawKeepSegments.filter((seg) => {
    if (seg.start >= seg.end) {
      logger.warn('Skipping invalid segment: start >= end', { start: seg.start, end: seg.end });
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

  // Stats
  const selectedByAI = keepIdSet.size;
  const keptWords = merged.words.filter((w) => keepIdSet.has(w.id));
  const flaggedSegmentCount = flaggedRanges.filter((r) => r.flagged).length;
  const aiCalls = usage2.inputTokens > 0 ? 2 : 1;

  // Coverage log
  const presenterWordCount = merged.words.filter((w) => w.is_presenter).length;
  if (presenterWordCount > 0) {
    const coverage = selectedByAI / presenterWordCount;
    logger.info('AI coverage', {
      selected: selectedByAI,
      presenter: presenterWordCount,
      coverage: `${(coverage * 100).toFixed(0)}%`,
    });
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
      take_selector_hints: takeDecisions.remove_ids.length,
      flagged_segments: flaggedSegmentCount,
      ai_calls: aiCalls,
      ai_brain: brain,
      ai_cost_usd: usage.estimatedCostUSD,
      processing_time_ms: processingTimeMs,
      sentence_menu: {
        total: sentences.length,
        selected: selectedIds.length,
        avg_score: sentenceData.stats.avg_score,
      },
    },
  };

  const outputPath = join(jobDir, 'cleaned_transcript.json');
  writeFileSync(outputPath, JSON.stringify(cleanedTranscript, null, 2), 'utf-8');

  logger.info('Merge and clean completed', {
    outputPath,
    originalWords: merged.stats.total_words,
    presenterWords: merged.stats.presenter_words,
    cleanedWords: keptWords.length,
    selectedByAI,
    sentencesTotal: sentences.length,
    sentencesSelected: selectedIds.length,
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
      selectedByAI,
      sentencesTotal: sentences.length,
      sentencesSelected: selectedIds.length,
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
