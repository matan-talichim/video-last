import { existsSync, readFileSync, writeFileSync, mkdirSync, unlinkSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { registerStep } from '../engine.js';
import type { StepContext, StepResult } from '../types.js';
import type { FFmpegConfig } from '../../utils/config.js';
import { loadConfig } from '../../utils/config.js';
import { runFFmpeg, getMediaInfo } from '../../utils/ffmpeg.js';
import type { Logger } from '../../utils/logger.js';

// ── Types ────────────────────────────────────────

interface TranscriptKeepSegment {
  start: number;
  end: number;
  [key: string]: unknown;
}

interface TranscriptWord {
  start: number;
  end: number;
  [key: string]: unknown;
}

interface CleanedTranscript {
  keep_segments: TranscriptKeepSegment[];
  words?: TranscriptWord[];
  [key: string]: unknown;
}

interface KeepSegment {
  index: number;
  start: number;
  end: number;
  type: string;
}

interface EditResult {
  outputPath: string;
  originalDuration: number;
  editedDuration: number;
  segmentsKept: number;
  segmentsRemoved: number;
  compressionRatio: string;
}

// ── Config defaults (read from config, fallback to hardcoded) ──

function getEditAssemblyConfig() {
  const config = loadConfig();
  const ea = (config as unknown as Record<string, unknown>).editAssembly as {
    fadeDuration?: number;
    crf?: number;
    preset?: string;
    audioBitrate?: string;
    minSegmentDuration?: number;
  } | undefined;
  return {
    fadeDuration: ea?.fadeDuration ?? 0.03,
    crf: ea?.crf ?? 18,
    preset: ea?.preset ?? 'medium',
    audioBitrate: ea?.audioBitrate ?? '192k',
    minSegmentDuration: ea?.minSegmentDuration ?? 0.1,
  };
}

// Backward-compatible constants (used throughout the file)
const _eaConfig = getEditAssemblyConfig();
const DEFAULT_FADE_DURATION = _eaConfig.fadeDuration;
const DEFAULT_CRF = _eaConfig.crf;
const DEFAULT_PRESET = _eaConfig.preset;
const DEFAULT_AUDIO_BITRATE = _eaConfig.audioBitrate;
const MIN_SEGMENT_DURATION = _eaConfig.minSegmentDuration;

// ── Helpers ──────────────────────────────────────

function readStatus(statusPath: string): Record<string, unknown> {
  if (existsSync(statusPath)) {
    return JSON.parse(readFileSync(statusPath, 'utf-8')) as Record<string, unknown>;
  }
  return {};
}

function writeStatus(statusPath: string, data: Record<string, unknown>): void {
  writeFileSync(statusPath, JSON.stringify(data, null, 2), 'utf-8');
}

function mergeOverlapping(segments: KeepSegment[]): KeepSegment[] {
  if (segments.length <= 1) return segments;

  const sorted = [...segments].sort((a, b) => a.start - b.start);
  const merged: KeepSegment[] = [sorted[0]!];

  for (let i = 1; i < sorted.length; i++) {
    const current = sorted[i]!;
    const last = merged[merged.length - 1]!;

    if (current.start <= last.end) {
      last.end = Math.max(last.end, current.end);
      last.type = `${last.type}+${current.type}`;
    } else {
      merged.push(current);
    }
  }

  return merged;
}

function getSmartEnd(segmentEnd: number, paddingEnd: number, allWords: TranscriptWord[]): number {
  const rawEnd = segmentEnd + paddingEnd;

  // Find any word that STARTS between segmentEnd and rawEnd
  const nextWord = allWords.find(w =>
    w.start > segmentEnd && w.start < rawEnd
  );

  if (nextWord) {
    // Don't bleed into the next word — cut just before it
    return Math.max(segmentEnd, nextWord.start - 0.02);
  }

  return rawEnd;
}

function getSmartStart(segmentStart: number, paddingStart: number, allWords: TranscriptWord[]): number {
  const rawStart = segmentStart - paddingStart;

  // Find the last word that ENDS between rawStart and segmentStart
  let prevWord: TranscriptWord | undefined;
  for (let i = allWords.length - 1; i >= 0; i--) {
    const w = allWords[i]!;
    if (w.end < segmentStart && w.end > rawStart) {
      prevWord = w;
      break;
    }
  }

  if (prevWord) {
    // Don't bleed into the previous word — cut just after it
    return Math.min(segmentStart, prevWord.end + 0.02);
  }

  return rawStart;
}

// ── Main export ──────────────────────────────────

export async function runEditAssembly(
  jobDir: string,
  ffmpegConfig: FFmpegConfig,
  logger: Logger,
): Promise<void> {
  const startTime = Date.now();
  const statusPath = resolve(jobDir, 'status.json');
  const tempDir = resolve(jobDir, 'temp');
  const timestamp = Date.now();

  // Read padding from config
  const config = loadConfig();
  const editConfig = (config as unknown as Record<string, unknown>).editAssembly as
    { paddingStart?: number; paddingEnd?: number; denoiseNoiseFloor?: number; audioCrossfade?: number; loudnormIntegrated?: number; loudnormTruePeak?: number; loudnormRange?: number } | undefined;
  const paddingStart = editConfig?.paddingStart ?? 0.03;
  const paddingEnd = editConfig?.paddingEnd ?? 0.15;

  // Check for gated audio (noise-filtered)
  const gatedAudioPath = resolve(jobDir, 'audio_gated.wav');
  const useGatedAudio = existsSync(gatedAudioPath);

  logger.info('Edit assembly started', { jobDir, paddingStart, paddingEnd, useGatedAudio });

  // Update status to editing
  const currentStatus = readStatus(statusPath);
  const currentProgress = (currentStatus.progress ?? {}) as Record<string, unknown>;
  writeStatus(statusPath, {
    ...currentStatus,
    status: 'editing',
    progress: {
      ...currentProgress,
      editAssembly: { status: 'processing' },
    },
  });

  try {
    // ── Step 1: Read cleaned transcript & build keep segments ──

    const transcriptPath = resolve(jobDir, 'cleaned_transcript.json');
    if (!existsSync(transcriptPath)) {
      throw new Error('cleaned_transcript.json not found — run transcript cleaning before editing');
    }

    const cleanedTranscript = JSON.parse(
      readFileSync(transcriptPath, 'utf-8'),
    ) as CleanedTranscript;

    if (!cleanedTranscript.keep_segments?.length) {
      throw new Error('cleaned_transcript has no keep_segments');
    }

    // Quality Gate check — block edit if quality gate failed
    const quality = (cleanedTranscript as Record<string, unknown>).quality as
      { passed?: boolean; blocks?: string[]; warnings?: string[] } | undefined;
    if (quality && quality.passed === false) {
      const blockReasons = (quality.blocks ?? []).join('; ');
      throw new Error(`Quality Gate BLOCKED edit-assembly: ${blockReasons}`);
    }
    if (quality?.warnings?.length) {
      logger.warn('Quality Gate warnings present, proceeding with edit', {
        warnings: quality.warnings,
      });
    }

    // Extract word-level timestamps for smart padding
    const allWords: TranscriptWord[] = cleanedTranscript.words ?? [];
    logger.info('Words loaded for smart padding', { wordCount: allWords.length });

    // Find original video file
    const originalFiles = ['original.mov', 'original.mp4', 'original.avi', 'original.mkv', 'original.webm'];
    let originalPath = '';
    for (const f of originalFiles) {
      const p = resolve(jobDir, f);
      if (existsSync(p)) {
        originalPath = p;
        break;
      }
    }
    if (!originalPath) {
      throw new Error('Original video file not found');
    }

    // Get video duration
    const mediaInfo = await getMediaInfo(originalPath, ffmpegConfig, logger);
    const videoDuration = mediaInfo.duration;

    logger.info('Original video info', {
      duration: videoDuration,
      codec: mediaInfo.codec,
      resolution: `${mediaInfo.width}x${mediaInfo.height}`,
    });

    // Build keep segments from cleaned transcript
    const rawSegments: KeepSegment[] = cleanedTranscript.keep_segments.map((seg, i) => ({
      index: i,
      start: seg.start,
      end: seg.end,
      type: String(seg.type ?? 'keep'),
    }));

    // Sort chronologically and merge overlaps
    const keepSegments = mergeOverlapping(rawSegments);

    // Filter out segments that are too short
    const validSegments = keepSegments.filter((seg) => {
      const duration = seg.end - seg.start;
      if (duration < MIN_SEGMENT_DURATION) {
        logger.warn('Skipping segment — too short', {
          type: seg.type,
          start: seg.start,
          end: seg.end,
          duration,
        });
        return false;
      }
      return true;
    });

    if (validSegments.length === 0) {
      throw new Error('No valid segments to keep after filtering');
    }

    logger.info('Keep segments built', {
      total: cleanedTranscript.keep_segments.length,
      afterMerge: keepSegments.length,
      afterFilter: validSegments.length,
    });

    // ── Step 2 & 3: Cut segments with padding + fade ──

    if (!existsSync(tempDir)) {
      mkdirSync(tempDir, { recursive: true });
    }

    const segmentFiles: string[] = [];

    for (let i = 0; i < validSegments.length; i++) {
      const seg = validSegments[i]!;

      // Apply smart padding — avoid cutting into adjacent words
      const actualStart = Math.max(0, getSmartStart(seg.start, paddingStart, allWords));
      const actualEnd = Math.min(videoDuration, getSmartEnd(seg.end, paddingEnd, allWords));
      const segDuration = actualEnd - actualStart;

      const segmentFile = join(tempDir, `segment_${timestamp}_${String(i).padStart(3, '0')}.mp4`);

      logger.info(`Cutting segment ${i + 1}/${validSegments.length}`, {
        type: seg.type,
        start: actualStart,
        end: actualEnd,
        duration: segDuration,
      });

      // Build audio filter chain: denoise → loudness normalization → fade-in → fade-out
      const denoiseNf = editConfig?.denoiseNoiseFloor ?? -20;
      const denoise = `afftdn=nf=${denoiseNf}:tn=1:om=o`;
      const loudnormI = editConfig?.loudnormIntegrated ?? -16;
      const loudnormTP = editConfig?.loudnormTruePeak ?? -1.5;
      const loudnormLRA = editConfig?.loudnormRange ?? 11;
      const loudnorm = `loudnorm=I=${loudnormI}:TP=${loudnormTP}:LRA=${loudnormLRA}`;
      // Cap fade duration to 30% of segment to avoid overlap on short segments
      const effectiveFade = Math.min(DEFAULT_FADE_DURATION, segDuration * 0.3);
      const fadeIn = `afade=t=in:st=0:d=${effectiveFade}`;
      const fadeOut = `afade=t=out:st=${Math.max(0, segDuration - effectiveFade)}:d=${effectiveFade}`;
      const audioFilter = `${denoise},${loudnorm},${fadeIn},${fadeOut}`;

      // Use original fps (fallback to 30)
      const fps = mediaInfo.fps > 0 ? Math.round(mediaInfo.fps) : 30;

      const args: string[] = useGatedAudio
        ? [
            '-ss', String(actualStart),
            '-i', originalPath,
            '-ss', String(actualStart),
            '-i', gatedAudioPath,
            '-to', String(actualEnd - actualStart),
            '-map', '0:v',
            '-map', '1:a',
            '-c:v', 'libx264',
            '-preset', DEFAULT_PRESET,
            '-crf', String(DEFAULT_CRF),
            '-r', String(fps),
            '-af', audioFilter,
            '-c:a', 'aac',
            '-b:a', DEFAULT_AUDIO_BITRATE,
            '-avoid_negative_ts', 'make_zero',
            segmentFile,
          ]
        : [
            '-ss', String(actualStart),
            '-i', originalPath,
            '-to', String(actualEnd - actualStart),
            '-c:v', 'libx264',
            '-preset', DEFAULT_PRESET,
            '-crf', String(DEFAULT_CRF),
            '-r', String(fps),
            '-af', audioFilter,
            '-c:a', 'aac',
            '-b:a', DEFAULT_AUDIO_BITRATE,
            '-avoid_negative_ts', 'make_zero',
            segmentFile,
          ];

      await runFFmpeg(args, ffmpegConfig, logger);
      segmentFiles.push(segmentFile);
    }

    logger.info('All segments cut', { count: segmentFiles.length });

    // ── Step 5: Concat ──

    const concatListPath = join(tempDir, `concat_list_${timestamp}.txt`);
    const concatContent = segmentFiles.map((f) => `file '${f}'`).join('\n');
    writeFileSync(concatListPath, concatContent, 'utf-8');

    const outputPath = resolve(jobDir, 'edited.mp4');
    const audioCrossfade = editConfig?.audioCrossfade ?? 0;

    if (audioCrossfade > 0 && segmentFiles.length >= 2) {
      // Audio crossfade: build filter_complex with acrossfade between segments
      logger.info('Concatenating with audio crossfade', { outputPath, crossfadeMs: audioCrossfade * 1000 });

      // Build input args
      const inputArgs: string[] = [];
      for (const f of segmentFiles) {
        inputArgs.push('-i', f);
      }

      // Build audio crossfade chain: [0:a][1:a] acrossfade → [a01], [a01][2:a] acrossfade → [a012], ...
      const filterParts: string[] = [];
      let prevLabel = '[0:a]';
      for (let i = 1; i < segmentFiles.length; i++) {
        const outLabel = `[a${i}]`;
        filterParts.push(`${prevLabel}[${i}:a]acrossfade=d=${audioCrossfade}:c1=tri:c2=tri${outLabel}`);
        prevLabel = outLabel;
      }

      // Video: just concat with stream copy approach via concat filter
      const videoInputs = segmentFiles.map((_, i) => `[${i}:v]`).join('');
      filterParts.push(`${videoInputs}concat=n=${segmentFiles.length}:v=1:a=0[vout]`);

      const filterComplex = filterParts.join(';');

      const concatArgs = [
        ...inputArgs,
        '-filter_complex', filterComplex,
        '-map', '[vout]',
        '-map', prevLabel,
        '-c:v', 'libx264',
        '-preset', DEFAULT_PRESET,
        '-crf', String(DEFAULT_CRF),
        '-c:a', 'aac',
        '-b:a', DEFAULT_AUDIO_BITRATE,
        outputPath,
      ];

      await runFFmpeg(concatArgs, ffmpegConfig, logger);
    } else {
      // Simple concat with stream copy (fast, no crossfade)
      logger.info('Concatenating segments', { outputPath });

      const concatArgs = [
        '-f', 'concat',
        '-safe', '0',
        '-i', concatListPath,
        '-c', 'copy',
        outputPath,
      ];

      await runFFmpeg(concatArgs, ffmpegConfig, logger);
    }

    // ── Step 6: Cleanup ──

    logger.info('Cleaning up temp files');
    for (const f of segmentFiles) {
      try { unlinkSync(f); } catch { /* ignore */ }
    }
    try { unlinkSync(concatListPath); } catch { /* ignore */ }

    // ── Step 7: Get edited video info & update status ──

    const editedInfo = await getMediaInfo(outputPath, ffmpegConfig, logger);
    const editedDuration = editedInfo.duration;
    const removedDuration = videoDuration - editedDuration;
    const compressionRatio = editedDuration > 0
      ? `${(videoDuration / editedDuration).toFixed(1)}:1`
      : 'N/A';

    const editResult: EditResult = {
      outputPath: 'edited.mp4',
      originalDuration: Math.round(videoDuration * 100) / 100,
      editedDuration: Math.round(editedDuration * 100) / 100,
      segmentsKept: validSegments.length,
      segmentsRemoved: cleanedTranscript.keep_segments.length - validSegments.length,
      compressionRatio,
    };

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

    const updatedStatus = readStatus(statusPath);
    const updatedProgress = (updatedStatus.progress ?? {}) as Record<string, unknown>;
    writeStatus(statusPath, {
      ...updatedStatus,
      status: 'edited',
      progress: {
        ...updatedProgress,
        editAssembly: { status: 'done', duration: `${elapsed}s` },
      },
      editResult,
    });

    logger.info('Edit assembly completed', {
      editedDuration,
      originalDuration: videoDuration,
      removedDuration: Math.round(removedDuration * 100) / 100,
      compressionRatio,
      elapsed: `${elapsed}s`,
    });
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    logger.error('Edit assembly failed', { error: errorMsg });

    const failedStatus = readStatus(statusPath);
    const failedProgress = (failedStatus.progress ?? {}) as Record<string, unknown>;
    writeStatus(statusPath, {
      ...failedStatus,
      status: 'error',
      error: errorMsg,
      progress: {
        ...failedProgress,
        editAssembly: { status: 'error', error: errorMsg },
      },
    });

    throw err;
  }
}

// ── Pipeline-compatible wrapper ─────────────────

async function editAssembly(context: StepContext): Promise<StepResult> {
  const { outputDir, logger } = context;
  const config = loadConfig();

  try {
    await runEditAssembly(outputDir, config.ffmpeg, logger);
    return {
      outputFile: resolve(outputDir, 'edited.mp4'),
      success: true,
      message: 'Edit assembly completed',
    };
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    return {
      outputFile: context.currentFile,
      success: false,
      message: errorMsg,
    };
  }
}

registerStep('edit-assembly', editAssembly);
