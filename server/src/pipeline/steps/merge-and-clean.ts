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

interface RemoveRange {
  ids: number[];
  reason: string;
}

interface AICleanResult {
  remove_ranges: RemoveRange[];
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

  const args = [
    scriptPath,
    '--transcript', transcriptPath,
    '--segments', segmentsPath,
    '--output', outputPath,
    '--buffer', '0.25',
  ];

  logger.info('Running merge_transcript.py', { transcriptPath, segmentsPath });

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

// ── Step 2: Semantic cleanup (AI) ───────────────

function buildNumberedText(words: Word[]): string {
  const presenterWords = words.filter((w) => w.is_presenter);
  return presenterWords.map((w) => `[${w.id}] ${w.word}`).join(' ');
}

const CLEANUP_PROMPT = `You are cleaning a Hebrew video transcript for a marketing video editor.
You receive numbered words spoken by the presenter.
Your job: identify ONLY clear junk to remove. When in doubt — KEEP.

RULES:

1. DUPLICATE TAKES (CLEAR RETAKES ONLY):
   A duplicate take is when the presenter STOPS mid-sentence, pauses, and
   RESTARTS the same sentence from the beginning.
   - Only mark as duplicate if the sentence clearly restarts (same opening 2-3 words).
   - Keep the LAST (final) take — remove earlier abandoned attempts.
   - Do NOT remove content that covers different aspects of the same topic.
   - Do NOT remove a sentence just because it has similar words to another.

2. PRODUCTION INSTRUCTIONS:
   Remove words that are clearly crew/director instructions, not part of the script.
   Dictionary: "ממשיכים", "שוב", "טייק", "נתחיל", "עוד פעם", "מהתחלה",
   "שוב מנקודה זו", "עוצר", "פסול", "סליחה", "שקט מצלמים",
   "לא טוב", "אני עושה שוב", "בואו נעשה עוד אחד"

3. LINE FEEDING:
   If the producer whispers 2-4 words and then the presenter repeats
   those exact words immediately after — remove the producer's whisper (first occurrence).

4. GOLDEN RULE — WHEN IN DOUBT, KEEP:
   - Better to include a slightly imperfect moment than to cut important content.
   - If a sentence adds new information or a new angle — it is NOT a duplicate.
   - If you're not 100% sure it's junk — KEEP IT.
   - Your goal: clean transcript, not short transcript.

Return JSON with remove_ranges only for items you are 100% certain are junk.
{
  "remove_ranges": [
    { "ids": [48, 49, 50], "reason": "duplicate_take" },
    { "ids": [72, 73, 74, 75], "reason": "production_instruction" }
  ]
}

If nothing to remove, return: { "remove_ranges": [] }

Numbered text:
`;

async function runSemanticCleanup(
  jobDir: string,
  merged: MergedTranscript,
  brain: AIBrain,
  logger: Logger,
): Promise<{ cleanResult: AICleanResult; usage: AIUsage }> {
  const numberedText = buildNumberedText(merged.words);
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

  // Ensure remove_ranges is an array
  if (!Array.isArray(data.remove_ranges)) {
    data.remove_ranges = [];
  }

  const totalRemoved = data.remove_ranges.reduce((sum, r) => sum + r.ids.length, 0);
  const reasons: Record<string, number> = {};
  for (const r of data.remove_ranges) {
    reasons[r.reason] = (reasons[r.reason] ?? 0) + r.ids.length;
  }

  logger.info('Semantic cleanup completed', {
    removeRanges: data.remove_ranges.length,
    totalRemovedWords: totalRemoved,
    removalReasons: reasons,
  });

  return { cleanResult: data, usage };
}

// ── Step 3: Build keep/remove segments from words ──

function buildSegments(
  allWords: Word[],
  removeRanges: RemoveRange[],
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
    } else if (!w.is_presenter) {
      removedWords.push({ ...w, reason: 'non_presenter' });
    } else {
      keptWords.push(w);
    }
  }

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

  // Step 2: Semantic cleanup via AI
  const { cleanResult, usage } = await runSemanticCleanup(jobDir, merged, brain, logger);

  // Step 3: Build keep/remove segments from word-level data
  const { keepSegments: rawKeepSegments, removeSegments } = buildSegments(merged.words, cleanResult.remove_ranges);

  // Filter out micro-segments (< 0.8s) — no real sentence is that short
  const MIN_KEEP_SEGMENT_DURATION = 0.8;
  const keepSegments = rawKeepSegments.filter((seg) => {
    const duration = seg.end - seg.start;
    if (duration < MIN_KEEP_SEGMENT_DURATION) {
      logger.info('Removing micro-segment', { start: seg.start, end: seg.end, duration: Math.round(duration * 1000) / 1000 });
      return false;
    }
    return true;
  });

  const processingTimeMs = Date.now() - startTime;

  // Count stats
  const removeIdSet = new Set<number>();
  for (const range of cleanResult.remove_ranges) {
    for (const id of range.ids) {
      removeIdSet.add(id);
    }
  }
  const keptWords = merged.words.filter((w) => w.is_presenter && !removeIdSet.has(w.id));

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
      removed_by_ai: removeIdSet.size,
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
    removedByAI: removeIdSet.size,
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
      removedByAI: removeIdSet.size,
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
