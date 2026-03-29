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

interface MergedTranscript {
  presenter_words: Array<{ word: string; start: number; end: number; confidence: number }>;
  other_words: Array<{ word: string; start: number; end: number; confidence: number; speaker: string }>;
  presenter_text: string;
  presenter_utterances: Array<{ text: string; start: number; end: number }>;
  stats: {
    total_words: number;
    presenter_words: number;
    other_words: number;
    filter_ratio: number;
    processing_time_ms: number;
  };
}

interface CleanedUtterance {
  text: string;
  start: number;
  end: number;
  action: 'keep' | 'remove';
  reason?: string;
}

interface AICleanResult {
  cleaned_text: string;
  cleaned_utterances: CleanedUtterance[];
  removed_count: number;
  removal_reasons: Record<string, number>;
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

const CLEANUP_PROMPT = `קיבלת תמלול של חומרי גלם מסרטון פרזנטור. התמלול כבר עבר סינון ראשוני אבל עדיין עלולים להופיע בו:

1. טייקים חוזרים — הפרזנטור אמר משפט ואז חזר עליו. השאר רק את הטייק האחרון (בדרך כלל הטוב יותר).
2. הוראות הפקה — ביטויים כמו "נתחיל שוב", "מוכן?", "עוד פעם", "רגע רגע", "מההתחלה", "שנייה", "go", "תתחיל", "מצלמה רצה", "יאללה" — מחק אותם.
3. מילות פתיחה מיותרות — "אוקיי אז", "טוב אז", "יאללה אז" בתחילת משפטים — מחק.

החזר JSON בפורמט הבא:
{
  "cleaned_text": "הטקסט הנקי",
  "cleaned_utterances": [
    { "text": "...", "start": 0.52, "end": 4.50, "action": "keep" },
    { "text": "...", "start": 5.10, "end": 6.20, "action": "remove", "reason": "duplicate take" },
    { "text": "...", "start": 7.00, "end": 7.50, "action": "remove", "reason": "production instruction" }
  ],
  "removed_count": 5,
  "removal_reasons": { "duplicate_take": 3, "production_instruction": 2 }
}

תמלול:
`;

async function runSemanticCleanup(
  jobDir: string,
  merged: MergedTranscript,
  brain: AIBrain,
  logger: Logger,
): Promise<{ cleanResult: AICleanResult; usage: AIUsage }> {
  // Build prompt with utterances and timestamps
  const utteranceLines = merged.presenter_utterances.map(
    (u) => `[${u.start.toFixed(2)}-${u.end.toFixed(2)}] ${u.text}`,
  );
  const userPrompt = CLEANUP_PROMPT + utteranceLines.join('\n');

  const aiConfig = loadConfig();
  const aiSettings = (aiConfig as unknown as Record<string, unknown>).ai as
    { timeout?: number; maxTokens?: number } | undefined;

  logger.info('Starting semantic cleanup', { brain, utteranceCount: merged.presenter_utterances.length });

  const { data, usage } = await askAIJSON<AICleanResult>(userPrompt, {
    brain,
    maxTokens: aiSettings?.maxTokens ?? 4096,
    timeout: aiSettings?.timeout ?? 60000,
    logger,
  });

  logger.info('Semantic cleanup completed', {
    removedCount: data.removed_count,
    removalReasons: data.removal_reasons,
  });

  return { cleanResult: data, usage };
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

  const processingTimeMs = Date.now() - startTime;

  // Build keep/remove segments from cleaned_utterances
  const keepSegments = cleanResult.cleaned_utterances
    .filter((u) => u.action === 'keep')
    .map((u) => ({ start: u.start, end: u.end }));

  const removeSegments = cleanResult.cleaned_utterances
    .filter((u) => u.action === 'remove')
    .map((u) => ({ start: u.start, end: u.end, reason: u.reason ?? 'unknown' }));

  // Count cleaned words (approximate from cleaned_text)
  const cleanedWords = cleanResult.cleaned_text.split(/\s+/).filter(Boolean).length;

  // Build final output
  const cleanedTranscript = {
    cleaned_text: cleanResult.cleaned_text,
    keep_segments: keepSegments,
    remove_segments: removeSegments,
    stats: {
      original_words: merged.stats.total_words,
      presenter_words: merged.stats.presenter_words,
      cleaned_words: cleanedWords,
      removed_by_merge: merged.stats.other_words,
      removed_by_ai: cleanResult.removed_count,
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
    cleanedWords,
    removedByMerge: merged.stats.other_words,
    removedByAI: cleanResult.removed_count,
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
      cleanedWords,
      removedByMerge: merged.stats.other_words,
      removedByAI: cleanResult.removed_count,
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
