import { copyFileSync } from 'node:fs';
import { basename, join } from 'node:path';
import { registerStep } from '../engine.js';
import type { StepContext, StepResult } from '../types.js';

async function passthrough(context: StepContext): Promise<StepResult> {
  const { currentFile, outputDir, logger } = context;
  const startTime = Date.now();

  logger.info('Passthrough step: received file', { currentFile });

  const outputFile = join(outputDir, basename(currentFile));

  if (currentFile !== outputFile) {
    copyFileSync(currentFile, outputFile);
    logger.info('Passthrough step: copied file to output', { outputFile });
  } else {
    logger.info('Passthrough step: file already in output directory');
  }

  const durationMs = Date.now() - startTime;
  logger.info('Passthrough step: completed', { durationMs });

  return {
    outputFile,
    success: true,
    message: `File passed through in ${durationMs}ms`,
  };
}

registerStep('passthrough', passthrough);
