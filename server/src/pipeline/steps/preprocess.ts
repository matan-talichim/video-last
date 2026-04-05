import { mkdirSync, existsSync, writeFileSync, readFileSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { registerStep } from '../engine.js';
import type { StepContext, StepResult } from '../types.js';
import { runFFmpeg } from '../../utils/ffmpeg.js';
import type { FFmpegConfig } from '../../utils/config.js';
import type { Logger } from '../../utils/logger.js';

// ── Types ────────────────────────────────────────

type SubStepStatus = 'pending' | 'processing' | 'done' | 'error';

interface SubStepProgress {
  status: SubStepStatus;
  durationMs?: number;
  error?: string;
}

interface PreprocessStatus {
  status: 'processing' | 'preprocessed' | 'error';
  currentStep: 'extracting_audio' | 'creating_proxy' | 'extracting_frames' | 'done';
  progress: {
    audio: SubStepProgress;
    proxy: SubStepProgress;
    frames: SubStepProgress;
  };
  preprocess?: {
    audioPath: string;
    proxyPath: string;
    framesDir: string;
    framesCount: number;
    duration: number;
  };
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

function writeStatus(statusPath: string, preprocessStatus: PreprocessStatus): void {
  const current = readStatusFile(statusPath);
  const merged = { ...current, ...preprocessStatus };
  writeFileSync(statusPath, JSON.stringify(merged, null, 2), 'utf-8');
}

// ── Main preprocess function (exported for direct use by API) ──

export async function runPreprocess(
  jobDir: string,
  inputFile: string,
  ffmpegConfig: FFmpegConfig,
  logger: Logger,
  options: { audioSampleRate: number; proxyHeight: number; proxyFps: number; frameInterval: number; audioScrub: { enabled: boolean; thresholdDb: number } },
): Promise<void> {
  const statusPath = join(jobDir, 'status.json');

  const preprocessStatus: PreprocessStatus = {
    status: 'processing',
    currentStep: 'extracting_audio',
    progress: {
      audio: { status: 'pending' },
      proxy: { status: 'pending' },
      frames: { status: 'pending' },
    },
  };

  writeStatus(statusPath, preprocessStatus);

  const audioPath = join(jobDir, 'audio.wav');
  const proxyPath = join(jobDir, 'proxy.mp4');
  const framesDir = join(jobDir, 'frames');

  let duration = 0;

  // ── Step 1: Extract audio ──────────────────────

  preprocessStatus.currentStep = 'extracting_audio';
  preprocessStatus.progress.audio.status = 'processing';
  writeStatus(statusPath, preprocessStatus);

  const audioStart = Date.now();
  try {
    const audioArgs = [
      '-i', inputFile,
      '-vn',
      '-acodec', 'pcm_s16le',
      '-ar', String(options.audioSampleRate),
      '-ac', '1',
      audioPath,
    ];
    logger.info('Preprocess: extracting audio', { command: `ffmpeg ${audioArgs.join(' ')}` });
    await runFFmpeg(audioArgs, ffmpegConfig, logger);

    const audioMs = Date.now() - audioStart;
    preprocessStatus.progress.audio = { status: 'done', durationMs: audioMs };
    logger.info('Preprocess: audio extraction completed', { durationMs: audioMs, outputPath: audioPath });

    // ── Step 1b: Create noise-gated audio ──────────
    const audioScrubConfig = options.audioScrub;
    if (audioScrubConfig.enabled) {
      const gatedPath = join(jobDir, 'audio_gated.wav');
      const gateStart = Date.now();
      try {
        const thresholdDb = audioScrubConfig.thresholdDb;
        const gateArgs = [
          '-i', audioPath,
          '-af', `agate=threshold=${thresholdDb}dB:ratio=6:attack=5:release=400`,
          gatedPath,
        ];
        logger.info('Preprocess: applying noise gate', {
          command: `ffmpeg ${gateArgs.join(' ')}`,
          thresholdDb,
        });
        await runFFmpeg(gateArgs, ffmpegConfig, logger);

        const gateMs = Date.now() - gateStart;
        logger.info('Preprocess: noise gate completed', { durationMs: gateMs, outputPath: gatedPath });
      } catch (gateErr) {
        const gateMs = Date.now() - gateStart;
        const gateErrMsg = gateErr instanceof Error ? gateErr.message : String(gateErr);
        logger.error('Preprocess: noise gate failed, transcription will use original audio', {
          durationMs: gateMs,
          error: gateErrMsg,
        });
      }
    } else {
      logger.info('Preprocess: audioScrub disabled, skipping noise gate');
    }
  } catch (err) {
    const audioMs = Date.now() - audioStart;
    const errorMsg = err instanceof Error ? err.message : String(err);
    preprocessStatus.progress.audio = { status: 'error', durationMs: audioMs, error: errorMsg };
    logger.error('Preprocess: audio extraction failed', { durationMs: audioMs, error: errorMsg });
  }
  writeStatus(statusPath, preprocessStatus);

  // ── Step 2: Create proxy video ─────────────────

  preprocessStatus.currentStep = 'creating_proxy';
  preprocessStatus.progress.proxy.status = 'processing';
  writeStatus(statusPath, preprocessStatus);

  const proxyStart = Date.now();
  try {
    const proxyArgs = [
      '-i', inputFile,
      '-vf', `scale=-2:${options.proxyHeight},fps=${options.proxyFps}`,
      '-c:v', 'libx264',
      '-preset', 'ultrafast',
      '-an',
      proxyPath,
    ];
    logger.info('Preprocess: creating proxy video', { command: `ffmpeg ${proxyArgs.join(' ')}` });
    await runFFmpeg(proxyArgs, ffmpegConfig, logger);

    const proxyMs = Date.now() - proxyStart;
    preprocessStatus.progress.proxy = { status: 'done', durationMs: proxyMs };
    logger.info('Preprocess: proxy video created', { durationMs: proxyMs, outputPath: proxyPath });
  } catch (err) {
    const proxyMs = Date.now() - proxyStart;
    const errorMsg = err instanceof Error ? err.message : String(err);
    preprocessStatus.progress.proxy = { status: 'error', durationMs: proxyMs, error: errorMsg };
    logger.error('Preprocess: proxy video creation failed', { durationMs: proxyMs, error: errorMsg });
  }
  writeStatus(statusPath, preprocessStatus);

  // ── Step 3: Extract frames ─────────────────────

  preprocessStatus.currentStep = 'extracting_frames';
  preprocessStatus.progress.frames.status = 'processing';
  writeStatus(statusPath, preprocessStatus);

  const framesStart = Date.now();
  try {
    if (!existsSync(framesDir)) {
      mkdirSync(framesDir, { recursive: true });
    }

    const framesArgs = [
      '-i', inputFile,
      '-vf', `fps=1/${options.frameInterval}`,
      '-q:v', '2',
      join(framesDir, 'frame_%04d.jpg'),
    ];
    logger.info('Preprocess: extracting frames', { command: `ffmpeg ${framesArgs.join(' ')}` });
    await runFFmpeg(framesArgs, ffmpegConfig, logger);

    const framesMs = Date.now() - framesStart;
    const framesCount = readdirSync(framesDir).filter((f) => f.endsWith('.jpg')).length;
    preprocessStatus.progress.frames = { status: 'done', durationMs: framesMs };
    logger.info('Preprocess: frames extracted', { durationMs: framesMs, framesCount, framesDir });

    // Get duration from media info if audio succeeded
    try {
      const { getMediaInfo } = await import('../../utils/ffmpeg.js');
      const mediaInfo = await getMediaInfo(inputFile, ffmpegConfig, logger);
      duration = mediaInfo.duration;
    } catch {
      logger.warn('Preprocess: could not get duration for metadata');
    }

    preprocessStatus.preprocess = {
      audioPath,
      proxyPath,
      framesDir,
      framesCount,
      duration,
    };
  } catch (err) {
    const framesMs = Date.now() - framesStart;
    const errorMsg = err instanceof Error ? err.message : String(err);
    preprocessStatus.progress.frames = { status: 'error', durationMs: framesMs, error: errorMsg };
    logger.error('Preprocess: frame extraction failed', { durationMs: framesMs, error: errorMsg });
  }

  // ── Final status ───────────────────────────────

  const hasErrors = preprocessStatus.progress.audio.status === 'error'
    && preprocessStatus.progress.proxy.status === 'error'
    && preprocessStatus.progress.frames.status === 'error';

  preprocessStatus.status = hasErrors ? 'error' : 'preprocessed';
  preprocessStatus.currentStep = 'done';
  writeStatus(statusPath, preprocessStatus);

  logger.info('Preprocess: all sub-steps completed', {
    audio: preprocessStatus.progress.audio.status,
    proxy: preprocessStatus.progress.proxy.status,
    frames: preprocessStatus.progress.frames.status,
  });
}

// ── Pipeline step wrapper ────────────────────────

async function preprocess(context: StepContext): Promise<StepResult> {
  const { originalFile, outputDir, logger, config } = context;

  const audioScrubRaw = config.audioScrub as { enabled?: boolean; thresholdDb?: number } | undefined;
  const options = {
    audioSampleRate: (config.audioSampleRate as number) ?? 16000,
    proxyHeight: (config.proxyHeight as number) ?? 480,
    proxyFps: (config.proxyFps as number) ?? 10,
    frameInterval: (config.frameInterval as number) ?? 5,
    audioScrub: {
      enabled: audioScrubRaw?.enabled ?? true,
      thresholdDb: audioScrubRaw?.thresholdDb ?? -32,
    },
  };

  const ffmpegConfig = (config.ffmpeg as import('../../utils/config.js').FFmpegConfig) ?? {
    path: 'ffmpeg',
    ffprobePath: 'ffprobe',
    defaultVideoCodec: 'libx264',
    defaultAudioCodec: 'aac',
    defaultPreset: 'medium',
  };

  await runPreprocess(outputDir, originalFile, ffmpegConfig, logger, options);

  return {
    outputFile: originalFile,
    success: true,
    message: 'Pre-processing completed',
  };
}

registerStep('preprocess', preprocess);
