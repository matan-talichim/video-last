import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import type { FFmpegConfig } from './config.js';
import type { Logger } from './logger.js';
import type { MediaInfo } from '../pipeline/types.js';

const execFileAsync = promisify(execFile);

export async function checkFFmpeg(config: FFmpegConfig, logger: Logger): Promise<boolean> {
  try {
    await execFileAsync(config.path, ['-version']);
    logger.info('FFmpeg found');
    return true;
  } catch {
    logger.error(`FFmpeg not found at path: ${config.path}`);
    return false;
  }
}

export async function getMediaInfo(
  filePath: string,
  config: FFmpegConfig,
  logger: Logger,
): Promise<MediaInfo> {
  logger.debug('Getting media info', { filePath });

  const args = [
    '-v', 'quiet',
    '-print_format', 'json',
    '-show_format',
    '-show_streams',
    filePath,
  ];

  const { stdout } = await execFileAsync(config.ffprobePath, args);
  const probe = JSON.parse(stdout) as {
    format?: { duration?: string; size?: string };
    streams?: Array<{
      codec_type?: string;
      codec_name?: string;
      width?: number;
      height?: number;
      r_frame_rate?: string;
    }>;
  };

  const videoStream = probe.streams?.find((s) => s.codec_type === 'video');
  const audioStream = probe.streams?.find((s) => s.codec_type === 'audio');

  const fpsStr = videoStream?.r_frame_rate ?? '0/1';
  const [num, den] = fpsStr.split('/').map(Number);
  const fps = den ? num! / den : 0;

  const info: MediaInfo = {
    filePath,
    duration: parseFloat(probe.format?.duration ?? '0'),
    width: videoStream?.width ?? 0,
    height: videoStream?.height ?? 0,
    fps,
    codec: videoStream?.codec_name ?? 'unknown',
    audioCodec: audioStream?.codec_name ?? 'unknown',
    fileSize: parseInt(probe.format?.size ?? '0', 10),
  };

  logger.info('Media info retrieved', info);
  return info;
}

export async function runFFmpeg(
  args: string[],
  config: FFmpegConfig,
  logger: Logger,
): Promise<string> {
  const fullArgs = ['-y', ...args];
  logger.debug('Running FFmpeg', { command: `${config.path} ${fullArgs.join(' ')}` });

  try {
    const { stdout, stderr } = await execFileAsync(config.path, fullArgs);
    logger.debug('FFmpeg completed', { stderr: stderr.slice(0, 500) });
    return stdout;
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    logger.error('FFmpeg failed', {
      command: `${config.path} ${fullArgs.join(' ')}`,
      error: errorMessage,
    });
    throw err;
  }
}

export async function copyVideo(
  input: string,
  output: string,
  config: FFmpegConfig,
  logger: Logger,
): Promise<void> {
  logger.info('Copying video without re-encode', { input, output });
  await runFFmpeg(['-i', input, '-c', 'copy', output], config, logger);
}
