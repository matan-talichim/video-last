import { existsSync, readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import sharp from 'sharp';
import { registerStep } from '../engine.js';
import type { StepContext, StepResult } from '../types.js';
import type { Logger } from '../../utils/logger.js';
import { askAIJSON } from '../../utils/ai-client.js';
import type { AIBrain, AIUsage } from '../../utils/ai-client.js';
import { loadConfig } from '../../utils/config.js';
import { getAnalyzerPrompt } from '../../prompts/analyzer-prompt.js';

// ── Types ────────────────────────────────────────

interface CleanedTranscript {
  cleaned_text: string;
  keep_segments: Array<{ start: number; end: number }>;
  remove_segments: Array<{ start: number; end: number; reason: string }>;
  stats: Record<string, unknown>;
}

interface PresenterSegments {
  segments: Array<{ start: number; end: number; confidence: number }>;
  total_speech_duration: number;
  total_video_duration: number;
  speech_ratio: number;
}

interface UserSettings {
  videoType?: string;
  targetDuration?: string;
  template?: string;
  aiBrain?: string;
}

interface AnalysisResult {
  summary: string;
  detectedGenre: string;
  targetAudience: string;
  viralityScore: number;
  retentionRisk: string;
  hook: Record<string, unknown>;
  structure: Record<string, unknown>;
  strongPoints: Array<Record<string, unknown>>;
  weakPoints: Array<Record<string, unknown>>;
  brollSuggestions: Array<Record<string, unknown>>;
  editingPlan: Record<string, unknown>;
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

async function loadFramesAsBase64(framesDir: string, maxFrames: number, logger: Logger): Promise<string[]> {
  if (!existsSync(framesDir)) {
    logger.warn('Frames directory not found, skipping vision input', { framesDir });
    return [];
  }

  const frameFiles = readdirSync(framesDir)
    .filter((f: string) => f.endsWith('.jpg') || f.endsWith('.jpeg') || f.endsWith('.png'))
    .sort();

  if (frameFiles.length === 0) {
    logger.warn('No frame files found', { framesDir });
    return [];
  }

  // Select evenly spaced frames
  const step = Math.max(1, Math.floor(frameFiles.length / maxFrames));
  const selected: string[] = [];
  for (let i = 0; i < frameFiles.length && selected.length < maxFrames; i += step) {
    const filePath = join(framesDir, frameFiles[i]!);
    const rawBuffer = readFileSync(filePath);

    // Compress to 480p max height, JPEG quality 65
    const compressed = await sharp(rawBuffer)
      .resize({ height: 480, withoutEnlargement: true })
      .jpeg({ quality: 65 })
      .toBuffer();

    selected.push(compressed.toString('base64'));
  }

  logger.info('Loaded and compressed frames for analysis', {
    totalFrames: frameFiles.length,
    selectedFrames: selected.length,
  });

  return selected;
}

// ── Main exported function ──────────────────────

export async function runAnalyze(
  jobDir: string,
  brain: AIBrain,
  logger: Logger,
): Promise<void> {
  const statusPath = join(jobDir, 'status.json');
  const startTime = Date.now();

  // Update status: analysis starting
  const currentStatus = readStatusFile(statusPath);
  const currentProgress = (currentStatus.progress as Record<string, unknown>) ?? {};
  writeStatus(statusPath, {
    currentStep: 'analysis',
    progress: {
      ...currentProgress,
      analysis: { status: 'processing' },
    },
  });

  // 1. Read cleaned transcript
  const cleanedPath = join(jobDir, 'cleaned_transcript.json');
  if (!existsSync(cleanedPath)) {
    throw new Error(`Cleaned transcript not found: ${cleanedPath}`);
  }
  const cleanedTranscript = JSON.parse(readFileSync(cleanedPath, 'utf-8')) as CleanedTranscript;

  // 2. Read presenter segments
  const segmentsPath = join(jobDir, 'presenter_segments.json');
  let presenterSegments: PresenterSegments | null = null;
  if (existsSync(segmentsPath)) {
    presenterSegments = JSON.parse(readFileSync(segmentsPath, 'utf-8')) as PresenterSegments;
  } else {
    logger.warn('Presenter segments not found, continuing without', { segmentsPath });
  }

  // 3. Read user settings
  const settingsPath = join(jobDir, 'settings.json');
  let userSettings: UserSettings = {};
  if (existsSync(settingsPath)) {
    userSettings = JSON.parse(readFileSync(settingsPath, 'utf-8')) as UserSettings;
  }

  // 4. Load frames (up to maxFrames, evenly spaced)
  const config = loadConfig();
  const analysisConfig = (config as unknown as Record<string, unknown>).analysis as
    { timeout?: number; maxFrames?: number } | undefined;
  const maxFrames = analysisConfig?.maxFrames ?? 10;
  const timeout = analysisConfig?.timeout ?? 120000;

  const framesDir = join(jobDir, 'frames');
  const frames = await loadFramesAsBase64(framesDir, maxFrames, logger);

  // 5. Build system prompt
  const systemPrompt = getAnalyzerPrompt({
    videoType: userSettings.videoType ?? 'general',
    targetDuration: userSettings.targetDuration ?? 'auto',
    template: userSettings.template ?? '',
  });

  // 6. Build user prompt
  let userPrompt = `הנה הנתונים של הסרטון:

## תמלול נקי:
${cleanedTranscript.cleaned_text}

## קטעי שמירה (timestamps):
${cleanedTranscript.keep_segments.map((s) => `[${s.start.toFixed(2)}-${s.end.toFixed(2)}]`).join(', ')}
`;

  if (presenterSegments) {
    userPrompt += `
## זמני דיבור של הפרזנטור:
${presenterSegments.segments.map((s) => `[${s.start.toFixed(2)}-${s.end.toFixed(2)}] (confidence: ${s.confidence.toFixed(2)})`).join('\n')}
משך דיבור כולל: ${presenterSegments.total_speech_duration.toFixed(1)} שניות
משך סרטון כולל: ${presenterSegments.total_video_duration.toFixed(1)} שניות
יחס דיבור: ${(presenterSegments.speech_ratio * 100).toFixed(1)}%
`;
  }

  userPrompt += `
## הגדרות המשתמש:
- סוג: ${userSettings.videoType ?? 'general'}
- משך רצוי: ${userSettings.targetDuration ?? 'auto'}
- תבנית: ${userSettings.template ?? 'ללא'}

נתח את הסרטון והחזר תוכנית עריכה מלאה ב-JSON.`;

  // 7. Send to AI
  logger.info('Starting AI analysis', {
    brain,
    transcriptLength: cleanedTranscript.cleaned_text.length,
    framesCount: frames.length,
    timeout,
  });

  const aiConfig = (config as unknown as Record<string, unknown>).ai as
    { maxTokens?: number } | undefined;

  let data: AnalysisResult;
  let usage: AIUsage;

  try {
    const result = await askAIJSON<AnalysisResult>(userPrompt, {
      brain,
      systemPrompt,
      maxTokens: aiConfig?.maxTokens ?? 4096,
      timeout,
      logger,
    });
    data = result.data;
    usage = result.usage;
  } catch (err) {
    const processingTimeMs = Date.now() - startTime;
    const errorMsg = err instanceof Error ? err.message : String(err);
    const isTimeout = errorMsg.includes('abort') || errorMsg.includes('timeout') || errorMsg.includes('ETIMEDOUT');

    logger.error('AI analysis failed', {
      brain,
      error: errorMsg,
      isTimeout,
      processingTimeMs,
    });

    // Update status to error
    const errStatus = readStatusFile(statusPath);
    const errProgress = (errStatus.progress as Record<string, unknown>) ?? {};
    writeStatus(statusPath, {
      status: 'error',
      error: isTimeout
        ? `Analysis timed out after ${Math.round(processingTimeMs / 1000)}s`
        : `Analysis failed: ${errorMsg}`,
      progress: {
        ...errProgress,
        analysis: { status: 'error', durationMs: processingTimeMs, error: errorMsg },
      },
    });

    throw err;
  }

  const processingTimeMs = Date.now() - startTime;

  // 8. Save analysis.json
  const analysisOutput = {
    ...data,
    metadata: {
      brain,
      inputTokens: usage.inputTokens,
      outputTokens: usage.outputTokens,
      costUSD: usage.estimatedCostUSD,
      processingTimeMs,
      framesAnalyzed: frames.length,
      timestamp: new Date().toISOString(),
    },
  };

  const outputPath = join(jobDir, 'analysis.json');
  writeFileSync(outputPath, JSON.stringify(analysisOutput, null, 2), 'utf-8');

  logger.info('Analysis completed', {
    outputPath,
    viralityScore: data.viralityScore,
    retentionRisk: data.retentionRisk,
    strongPoints: data.strongPoints?.length ?? 0,
    weakPoints: data.weakPoints?.length ?? 0,
    brollSuggestions: data.brollSuggestions?.length ?? 0,
    brain,
    inputTokens: usage.inputTokens,
    outputTokens: usage.outputTokens,
    costUSD: usage.estimatedCostUSD.toFixed(4),
    processingTimeMs,
  });

  // 9. Update status
  const finalStatus = readStatusFile(statusPath);
  const finalProgress = (finalStatus.progress as Record<string, unknown>) ?? {};

  const existingCosts = (finalStatus.aiCosts as {
    totalCalls?: number;
    totalInputTokens?: number;
    totalOutputTokens?: number;
    estimatedCostUSD?: number;
  }) ?? {};

  writeStatus(statusPath, {
    status: 'analyzed',
    currentStep: 'done',
    progress: {
      ...finalProgress,
      analysis: { status: 'done', durationMs: processingTimeMs },
    },
    analysis: {
      viralityScore: data.viralityScore,
      retentionRisk: data.retentionRisk,
      detectedGenre: data.detectedGenre,
      strongPointsCount: data.strongPoints?.length ?? 0,
      weakPointsCount: data.weakPoints?.length ?? 0,
      brollSuggestionsCount: data.brollSuggestions?.length ?? 0,
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

async function analyze(context: StepContext): Promise<StepResult> {
  const { outputDir, logger } = context;

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

  await runAnalyze(outputDir, brain, logger);

  return {
    outputFile: context.originalFile,
    success: true,
    message: 'Video analysis completed',
  };
}

registerStep('analyze', analyze);
