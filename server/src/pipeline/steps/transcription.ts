import { createReadStream, existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { DeepgramClient } from '@deepgram/sdk';
import { registerStep } from '../engine.js';
import type { StepContext, StepResult } from '../types.js';
import type { Logger } from '../../utils/logger.js';

// ── Types ────────────────────────────────────────

interface TranscriptionWord {
  word: string;
  start: number;
  end: number;
  confidence: number;
}

interface TranscriptionUtterance {
  text: string;
  start: number;
  end: number;
  confidence: number;
}

interface TranscriptionParagraph {
  text: string;
  start: number;
  end: number;
}

interface TranscriptionResult {
  text: string;
  words: TranscriptionWord[];
  utterances: TranscriptionUtterance[];
  paragraphs: TranscriptionParagraph[];
  metadata: {
    duration: number;
    language: string;
    model: string;
    wordCount: number;
    processingTimeMs: number;
  };
}

interface TranscriptionConfig {
  model: string;
  language: string;
  smartFormat: boolean;
  utterances: boolean;
  punctuate: boolean;
  words: boolean;
  paragraphs: boolean;
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

// ── Main transcription function ─────────────────

export async function runTranscription(
  jobDir: string,
  transcriptionConfig: TranscriptionConfig,
  logger: Logger,
): Promise<void> {
  const statusPath = join(jobDir, 'status.json');
  const gatedAudioPath = join(jobDir, 'audio_gated.wav');
  const originalAudioPath = join(jobDir, 'audio.wav');
  const audioPath = existsSync(gatedAudioPath) ? gatedAudioPath : originalAudioPath;
  logger.info('Transcription: selected audio file', {
    selectedPath: audioPath,
    gatedExists: existsSync(gatedAudioPath),
  });
  const rawOutputPath = join(jobDir, 'transcript_raw.json');
  const outputPath = join(jobDir, 'transcript.json');

  // Update status: transcription starting
  const currentStatus = readStatusFile(statusPath);
  const currentProgress = (currentStatus.progress as Record<string, unknown>) ?? {};
  writeStatus(statusPath, {
    currentStep: 'transcription',
    progress: {
      ...currentProgress,
      transcription: { status: 'processing' },
    },
  });

  // Validate API key
  const apiKey = process.env.DEEPGRAM_API_KEY;
  if (!apiKey) {
    throw new Error(
      'DEEPGRAM_API_KEY is not set. Please add it to your .env file.',
    );
  }

  // Validate input file exists
  if (!existsSync(audioPath)) {
    throw new Error(`Audio file not found: ${audioPath}`);
  }

  const startTime = Date.now();

  logger.info('Starting transcription', {
    jobDir,
    model: transcriptionConfig.model,
    language: transcriptionConfig.language,
  });

  // Initialize Deepgram client (v5 SDK)
  const deepgram = new DeepgramClient({ apiKey });

  // Send to Deepgram via v1 pre-recorded API
  const result = await deepgram.listen.v1.media.transcribeFile(
    createReadStream(audioPath),
    {
      model: transcriptionConfig.model,
      language: transcriptionConfig.language,
      smart_format: transcriptionConfig.smartFormat,
      utterances: transcriptionConfig.utterances,
      punctuate: transcriptionConfig.punctuate,
      paragraphs: transcriptionConfig.paragraphs,
    },
  );

  const processingTimeMs = Date.now() - startTime;

  // Save raw response
  writeFileSync(rawOutputPath, JSON.stringify(result, null, 2), 'utf-8');
  logger.info('Raw transcript saved', { path: rawOutputPath });

  // Narrow the union type — reject async callback responses
  if (!('results' in result)) {
    throw new Error('Deepgram returned an accepted-async response instead of results');
  }

  // Process the response into clean format
  const channels = result.results?.channels;
  const channel = channels?.[0];
  const alternative = channel?.alternatives?.[0];

  if (!alternative) {
    throw new Error('Deepgram returned no transcription alternatives');
  }

  const fullText = (alternative as Record<string, unknown>).transcript as string ?? '';

  // Extract words with timestamps
  const rawWords = (alternative as Record<string, unknown>).words as Array<Record<string, unknown>> ?? [];
  const words: TranscriptionWord[] = rawWords.map((w) => ({
    word: w.word as string,
    start: w.start as number,
    end: w.end as number,
    confidence: w.confidence as number,
  }));

  // Extract utterances
  const rawUtterances = (result.results as unknown as Record<string, unknown>)?.utterances as Array<Record<string, unknown>> ?? [];
  const utterances: TranscriptionUtterance[] = rawUtterances.map((u) => ({
    text: (u.transcript as string) ?? '',
    start: u.start as number,
    end: u.end as number,
    confidence: u.confidence as number,
  }));

  // Extract paragraphs
  const paragraphs: TranscriptionParagraph[] = [];
  const paragraphsData = (alternative as Record<string, unknown>).paragraphs as
    | { paragraphs?: Array<{ sentences?: Array<{ text: string; start: number; end: number }>; start?: number; end?: number }> }
    | undefined;

  if (paragraphsData?.paragraphs) {
    for (const p of paragraphsData.paragraphs) {
      const sentences = p.sentences ?? [];
      const text = sentences.map((s) => s.text).join(' ');
      const pStart = p.start ?? sentences[0]?.start ?? 0;
      const pEnd = p.end ?? sentences[sentences.length - 1]?.end ?? 0;
      paragraphs.push({ text, start: pStart, end: pEnd });
    }
  }

  // Build clean transcript
  const duration = result.metadata?.duration ?? 0;

  const transcript: TranscriptionResult = {
    text: fullText,
    words,
    utterances,
    paragraphs,
    metadata: {
      duration,
      language: transcriptionConfig.language,
      model: transcriptionConfig.model,
      wordCount: words.length,
      processingTimeMs,
    },
  };

  // Save clean transcript
  writeFileSync(outputPath, JSON.stringify(transcript, null, 2), 'utf-8');

  logger.info('Transcription completed', {
    wordCount: words.length,
    utteranceCount: utterances.length,
    paragraphCount: paragraphs.length,
    duration,
    processingTimeMs,
  });

  // Update status
  const finalStatus = readStatusFile(statusPath);
  const finalProgress = (finalStatus.progress as Record<string, unknown>) ?? {};
  writeStatus(statusPath, {
    status: 'transcribed',
    currentStep: 'done',
    progress: {
      ...finalProgress,
      transcription: { status: 'done', durationMs: processingTimeMs },
    },
    transcription: {
      wordCount: words.length,
      utteranceCount: utterances.length,
      paragraphCount: paragraphs.length,
      duration,
      outputPath,
    },
  });
}

// ── Pipeline step wrapper ────────────────────────

async function transcription(context: StepContext): Promise<StepResult> {
  const { outputDir, logger, config } = context;

  const transcriptionConfig: TranscriptionConfig = {
    model: (config.model as string) ?? 'nova-3',
    language: (config.language as string) ?? 'he',
    smartFormat: (config.smartFormat as boolean) ?? true,
    utterances: (config.utterances as boolean) ?? true,
    punctuate: (config.punctuate as boolean) ?? true,
    words: (config.words as boolean) ?? true,
    paragraphs: (config.paragraphs as boolean) ?? true,
  };

  await runTranscription(outputDir, transcriptionConfig, logger);

  return {
    outputFile: context.originalFile,
    success: true,
    message: 'Transcription completed',
  };
}

registerStep('transcription', transcription);
