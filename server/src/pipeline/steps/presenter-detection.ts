import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { execFile } from 'node:child_process';
import { registerStep } from '../engine.js';
import type { StepContext, StepResult } from '../types.js';
import type { Logger } from '../../utils/logger.js';

// ── Types ────────────────────────────────────────

interface PresenterSegment {
  start: number;
  end: number;
  confidence: number;
}

interface PresenterDetectionResult {
  segments: PresenterSegment[];
  total_speech_duration: number;
  total_video_duration: number;
  speech_ratio: number;
  processing_time_ms: number;
}

interface PresenterDetectionConfig {
  lipThreshold: number;
  vadThreshold: number;
  buffer: number;
  pythonPath: string;
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

function runCommand(
  command: string,
  args: string[],
  logger: Logger,
  timeoutMs: number = 120000,
): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    logger.info('Running external command', { command, args: args.join(' ') });

    const child = execFile(command, args, {
      timeout: timeoutMs,
      maxBuffer: 10 * 1024 * 1024, // 10MB
    }, (error, stdout, stderr) => {
      if (stderr) {
        // MediaPipe and torch write warnings to stderr — log as debug, not error
        logger.debug('Command stderr output', { stderr: stderr.slice(0, 2000) });
      }

      if (error) {
        reject(new Error(`Command failed: ${error.message}\nstderr: ${stderr}`));
        return;
      }

      resolve({ stdout, stderr });
    });

    child.on('error', (err) => {
      reject(new Error(`Failed to start command: ${err.message}`));
    });
  });
}

async function ensureVenv(logger: Logger): Promise<string> {
  const scriptDir = resolve('server/python');
  const venvPython = join(scriptDir, 'venv', 'bin', 'python3');

  if (existsSync(venvPython)) {
    return venvPython;
  }

  logger.info('Python venv not found, running setup.sh...');
  const setupScript = join(scriptDir, 'setup.sh');

  if (!existsSync(setupScript)) {
    throw new Error(`Setup script not found: ${setupScript}`);
  }

  await runCommand('bash', [setupScript], logger, 300000); // 5 min timeout for setup

  if (!existsSync(venvPython)) {
    throw new Error('setup.sh completed but venv python not found');
  }

  logger.info('Python venv created successfully');
  return venvPython;
}

// ── Main presenter detection function ───────────

export async function runPresenterDetection(
  jobDir: string,
  presenterConfig: PresenterDetectionConfig,
  logger: Logger,
): Promise<void> {
  const statusPath = join(jobDir, 'status.json');
  const audioPath = join(jobDir, 'audio.wav');
  const videoPath = join(jobDir, 'proxy.mp4');
  const outputPath = join(jobDir, 'presenter_segments.json');

  // Update status: presenter detection starting
  writeStatus(statusPath, {
    currentStep: 'presenter_detection',
    progress: {
      ...((readStatusFile(statusPath) as { progress?: Record<string, unknown> }).progress ?? {}),
      presenterDetection: { status: 'processing' },
    },
  });

  // Validate input files exist
  if (!existsSync(audioPath)) {
    throw new Error(`Audio file not found: ${audioPath}`);
  }
  if (!existsSync(videoPath)) {
    throw new Error(`Proxy video not found: ${videoPath}`);
  }

  const startTime = Date.now();

  // Ensure Python venv is ready
  let pythonPath: string;
  try {
    pythonPath = await ensureVenv(logger);
  } catch (err) {
    // Fallback to system python if configured
    if (presenterConfig.pythonPath !== 'python3') {
      pythonPath = presenterConfig.pythonPath;
    } else {
      throw err;
    }
  }

  const scriptPath = resolve('server/python/presenter_detection.py');
  if (!existsSync(scriptPath)) {
    throw new Error(`Python script not found: ${scriptPath}`);
  }

  // Build command arguments
  const args = [
    scriptPath,
    '--audio', audioPath,
    '--video', videoPath,
    '--output', outputPath,
    '--lip-threshold', String(presenterConfig.lipThreshold),
    '--vad-threshold', String(presenterConfig.vadThreshold),
    '--buffer', String(presenterConfig.buffer),
  ];

  logger.info('Running presenter detection', {
    pythonPath,
    args: args.join(' '),
    jobDir,
  });

  // Run Python script
  await runCommand(pythonPath, args, logger, 120000);

  // Read and validate output
  if (!existsSync(outputPath)) {
    throw new Error('Presenter detection completed but output file not found');
  }

  const resultJson = readFileSync(outputPath, 'utf-8');
  const result = JSON.parse(resultJson) as PresenterDetectionResult;

  const durationMs = Date.now() - startTime;

  logger.info('Presenter detection completed', {
    segments: result.segments.length,
    totalSpeech: result.total_speech_duration,
    speechRatio: result.speech_ratio,
    processingTimeMs: durationMs,
  });

  // Update status
  const currentStatus = readStatusFile(statusPath);
  const currentProgress = (currentStatus.progress as Record<string, unknown>) ?? {};
  writeStatus(statusPath, {
    status: 'presenter_detected',
    currentStep: 'done',
    progress: {
      ...currentProgress,
      presenterDetection: { status: 'done', durationMs },
    },
    presenterDetection: {
      segmentsCount: result.segments.length,
      totalSpeechDuration: result.total_speech_duration,
      totalVideoDuration: result.total_video_duration,
      speechRatio: result.speech_ratio,
      outputPath,
    },
  });
}

// ── Pipeline step wrapper ────────────────────────

async function presenterDetection(context: StepContext): Promise<StepResult> {
  const { outputDir, logger, config } = context;

  const presenterConfig: PresenterDetectionConfig = {
    lipThreshold: (config.lipThreshold as number) ?? 0.15,
    vadThreshold: (config.vadThreshold as number) ?? 0.5,
    buffer: (config.buffer as number) ?? 0.25,
    pythonPath: (config.pythonPath as string) ?? 'python3',
  };

  await runPresenterDetection(outputDir, presenterConfig, logger);

  return {
    outputFile: context.originalFile,
    success: true,
    message: 'Presenter detection completed',
  };
}

registerStep('presenter-detection', presenterDetection);
