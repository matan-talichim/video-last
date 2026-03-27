import { mkdirSync, existsSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import type {
  StepFunction,
  StepConfig,
  StepRunResult,
  StepContext,
  MediaInfo,
  PipelineMetadata,
} from './types.js';
import type { Logger } from '../utils/logger.js';
import type { AppConfig } from '../utils/config.js';
import { getMediaInfo } from '../utils/ffmpeg.js';

const stepRegistry = new Map<string, StepFunction>();

export function registerStep(name: string, fn: StepFunction): void {
  stepRegistry.set(name, fn);
}

export interface PipelineOptions {
  inputFile: string;
  outputDir: string;
  config: AppConfig;
  logger: Logger;
  userPrompt?: string;
}

export interface PipelineResult {
  success: boolean;
  outputFile: string;
  steps: StepRunResult[];
  totalDurationMs: number;
}

export async function runPipeline(options: PipelineOptions): Promise<PipelineResult> {
  const { inputFile, outputDir, config, logger, userPrompt } = options;
  const tempDir = join(outputDir, 'temp');
  const startTime = Date.now();
  const stepResults: StepRunResult[] = [];

  logger.info('Pipeline starting', { inputFile, outputDir });

  // Validate input
  if (!existsSync(inputFile)) {
    throw new Error(`Input file not found: ${inputFile}`);
  }

  // Create directories
  if (!existsSync(outputDir)) {
    mkdirSync(outputDir, { recursive: true });
  }
  if (!existsSync(tempDir)) {
    mkdirSync(tempDir, { recursive: true });
  }

  // Get media info
  const mediaInfo: MediaInfo = await getMediaInfo(inputFile, config.ffmpeg, logger);

  // Filter enabled steps
  const enabledSteps: StepConfig[] = config.pipeline.steps.filter((s) => s.enabled);
  logger.info(`Running ${enabledSteps.length} steps`);

  let currentFile = inputFile;
  const metadata: PipelineMetadata = {};

  // Run steps in sequence
  for (const stepConfig of enabledSteps) {
    const stepFn = stepRegistry.get(stepConfig.name);
    if (!stepFn) {
      logger.warn(`Step "${stepConfig.name}" not found in registry, skipping`);
      stepResults.push({
        stepName: stepConfig.name,
        success: false,
        durationMs: 0,
        error: 'Step not found in registry',
      });
      continue;
    }

    const stepStart = Date.now();
    logger.info(`Step "${stepConfig.name}" starting`);

    const context: StepContext = {
      currentFile,
      originalFile: inputFile,
      outputDir,
      tempDir,
      mediaInfo,
      metadata,
      config: stepConfig.options ?? {},
      logger,
      userPrompt,
    };

    try {
      const result = await stepFn(context);
      const durationMs = Date.now() - stepStart;

      if (result.success) {
        currentFile = result.outputFile;
        if (result.metadata) {
          Object.assign(metadata, result.metadata);
        }
        logger.info(`Step "${stepConfig.name}" completed`, { durationMs, message: result.message });
      } else {
        logger.warn(`Step "${stepConfig.name}" failed`, { durationMs, message: result.message });
      }

      stepResults.push({
        stepName: stepConfig.name,
        success: result.success,
        durationMs,
        message: result.message,
      });
    } catch (err) {
      const durationMs = Date.now() - stepStart;
      const errorMessage = err instanceof Error ? err.message : String(err);
      logger.error(`Step "${stepConfig.name}" threw an error`, { durationMs, error: errorMessage });

      stepResults.push({
        stepName: stepConfig.name,
        success: false,
        durationMs,
        error: errorMessage,
      });
    }
  }

  // Cleanup temp directory
  if (config.editor.cleanTempAfter && existsSync(tempDir)) {
    logger.info('Cleaning up temp directory');
    rmSync(tempDir, { recursive: true, force: true });
  }

  const totalDurationMs = Date.now() - startTime;
  logger.info('Pipeline completed', {
    totalDurationMs,
    stepsRun: stepResults.length,
    successful: stepResults.filter((s) => s.success).length,
  });

  return {
    success: stepResults.every((s) => s.success),
    outputFile: currentFile,
    steps: stepResults,
    totalDurationMs,
  };
}
