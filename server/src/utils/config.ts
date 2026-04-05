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

export interface AIConfig {
  defaultBrain: string;
  timeout: number;
  maxTokens: number;
}

export interface AnalysisConfig {
  timeout: number;
  maxFrames: number;
  maxTokens?: number;
}

export interface TranscriptionConfig {
  model: string;
  language: string;
  smartFormat: boolean;
  utterances: boolean;
  punctuate: boolean;
  words: boolean;
  paragraphs: boolean;
}

export interface PresenterDetectionConfig {
  lipThreshold: number;
  vadThreshold: number;
  buffer: number;
  pythonPath: string;
  speakerVerify: boolean;
}

export interface EditAssemblyConfig {
  paddingStart: number;
  paddingEnd: number;
  fadeDuration: number;
  crf: number;
  preset: string;
  audioBitrate: string;
  minSegmentDuration: number;
  denoiseNoiseFloor: number;
  audioCrossfade: number;
  loudnormIntegrated: number;
  loudnormTruePeak: number;
  loudnormRange: number;
}

export interface TakeSelectorConfig {
  enabled: boolean;
  similarityThreshold: number;
  lookbackSeconds: number;
  scoringOverrideMargin: number;
}

export interface WordScorerConfig {
  thresholdPresenter: number;
  thresholdReject: number;
}

export interface MergeAndCleanConfig {
  silenceThresholdMs: number;
  minSegmentAfterSplit: number;
  minKeepSegmentDuration: number;
}

export interface AudioScrubConfig {
  enabled: boolean;
  thresholdDb: number;
}

export interface AppConfig {
  editor: EditorConfig;
  server: { port: number };
  pipeline: { steps: StepConfigEntry[] };
  ffmpeg: FFmpegConfig;
  upload: UploadConfig;
  ai: AIConfig;
  analysis: AnalysisConfig;
  transcription: TranscriptionConfig;
  presenterDetection: PresenterDetectionConfig;
  editAssembly: EditAssemblyConfig;
  takeSelector: TakeSelectorConfig;
  wordScorer: WordScorerConfig;
  mergeAndClean: MergeAndCleanConfig;
  audioScrub: AudioScrubConfig;
}

export function loadConfig(configPath?: string): AppConfig {
  const filePath = configPath ?? resolve('config/default.json');
  const raw = readFileSync(filePath, 'utf-8');
  const config = JSON.parse(raw) as AppConfig;
  return config;
}
