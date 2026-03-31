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
    total_removed_words: number;
  };
  processing_time_ms: number;
}

interface AICleanResult {
  keep_ranges: KeepRange[];
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
  const args = [
    scriptPath,
    '--transcript', transcriptPath,
    '--segments', segmentsPath,
    '--output', outputPath,
    '--buffer', '0.25',
    '--audio', audioPath,
  ];

  logger.info('Running merge_transcript.py', { transcriptPath, segmentsPath, audioPath });

  const { stdout, stderr } = await execFileAsync(pythonPath, args, {
    timeout: 30000,
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
    timeout: 60000,
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
   - Words marked with * are non-presenter speech (background voice, crew,
     production assistant). Do NOT include them in your keep_ranges.
     The ONLY exception: if a single starred word (not a sequence of 2+ starred
     words) appears between two presenter words and is clearly the same speaker
     continuing the same sentence, you may include it.
     If you see 2 or more consecutive starred words — they are definitely
     non-presenter. NEVER include a sequence of starred words.
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

   - COMPLETE SENTENCES ONLY:
     Every segment you output MUST be a grammatically complete thought.
     - Do NOT start a segment mid-sentence (e.g., starting with "לך שהחיסכון...")
     - Do NOT end a segment with a hanging word (e.g., ending with "...לא")
     - If you can't find a complete version of a sentence, skip it entirely.
       A shorter video with complete sentences is ALWAYS better than a longer
       video with broken fragments.

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

async function runSemanticCleanup(
  jobDir: string,
  merged: MergedTranscript,
  brain: AIBrain,
  logger: Logger,
  takeSelectorIds?: Set<number>,
): Promise<{ cleanResult: AICleanResult; usage: AIUsage }> {
  const numberedText = buildNumberedText(merged.words, takeSelectorIds);
  const userPrompt = CLEANUP_PROMPT + numberedText;

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
  const merged = await runMerge(jobDir, logger);

  // Step 1.5: Take selector (rule-based duplicate detection)
  const takeDecisions = await runTakeSelector(jobDir, logger);
  const takeSelectorIds = new Set(takeDecisions.remove_ids);

  // Step 2: Semantic cleanup via AI (with take selector hints)
  const { cleanResult, usage } = await runSemanticCleanup(jobDir, merged, brain, logger, takeSelectorIds);

  // Build effective remove ranges from AI keep_ranges (invert: everything not kept is removed)
  const keepIdSet = new Set(cleanResult.keep_ranges.flatMap(r => r.ids));
  const effectiveRemoveRanges: KeepRange[] = [];
  if (keepIdSet.size > 0) {
    const notKeptIds = merged.words
      .filter(w => !keepIdSet.has(w.id))
      .map(w => w.id);
    if (notKeptIds.length > 0) {
      effectiveRemoveRanges.push({ ids: notKeptIds, reason: 'not_selected_by_ai' });
    }
  }

  // Step 3: Build keep/remove segments from word-level data
  const { keepSegments: rawKeepSegments, removeSegments } = buildSegments(merged.words, effectiveRemoveRanges);

  // Filter out invalid segments (start >= end) and micro-segments (< 0.8s)
  const MIN_KEEP_SEGMENT_DURATION = 0.8;
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
      keepSegments: keepSegments.length,
      brain,
      outputPath,
    },
    aiCosts: {
      totalCalls: (existingCosts.totalCalls ?? 0) + 1,
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
