import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

export interface EditorConfig {
  inputDir: string;
  outputDir: string;
  tempDir: string;
  logsDir: string;
  cleanTempAfter: boolean;
  logLevel: 'debug' | 'info' | 'warn' | 'error';
}

export interface FFmpegConfig {
  path: string;
  ffprobePath: string;
  defaultVideoCodec: string;
  defaultAudioCodec: string;
  defaultPreset: string;
}

export interface StepConfigEntry {
  name: string;
  enabled: boolean;
  description?: string;
  options?: Record<string, unknown>;
}

export interface UploadConfig {
  allowedFormats: string[];
  maxFileSize: number;
}

export interface AppConfig {
  editor: EditorConfig;
  server: { port: number };
  pipeline: { steps: StepConfigEntry[] };
  ffmpeg: FFmpegConfig;
  upload: UploadConfig;
}

export function loadConfig(configPath?: string): AppConfig {
  const filePath = configPath ?? resolve('config/default.json');
  const raw = readFileSync(filePath, 'utf-8');
  const config = JSON.parse(raw) as AppConfig;
  return config;
}
