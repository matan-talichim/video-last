export interface MediaInfo {
  filePath: string;
  duration: number;
  width: number;
  height: number;
  fps: number;
  codec: string;
  audioCodec: string;
  fileSize: number;
}

export interface PipelineMetadata {
  transcript?: string;
  speakers?: string[];
  [key: string]: unknown;
}

export interface StepContext {
  currentFile: string;
  originalFile: string;
  outputDir: string;
  tempDir: string;
  mediaInfo: MediaInfo;
  metadata: PipelineMetadata;
  config: Record<string, unknown>;
  logger: {
    debug: (message: string, data?: unknown) => void;
    info: (message: string, data?: unknown) => void;
    warn: (message: string, data?: unknown) => void;
    error: (message: string, data?: unknown) => void;
  };
  userPrompt?: string;
}

export interface StepResult {
  outputFile: string;
  metadata?: Partial<PipelineMetadata>;
  success: boolean;
  message?: string;
}

export type StepFunction = (context: StepContext) => Promise<StepResult>;

export interface StepConfig {
  name: string;
  enabled: boolean;
  description?: string;
  options?: Record<string, unknown>;
}

export interface StepRunResult {
  stepName: string;
  success: boolean;
  durationMs: number;
  message?: string;
  error?: string;
}
