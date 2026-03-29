import Anthropic from '@anthropic-ai/sdk';
import OpenAI from 'openai';
import type { Logger } from './logger.js';

// ── Types ────────────────────────────────────────

export type AIBrain = 'claude_sonnet_4_6' | 'gpt_5_4';

export interface AIClientOptions {
  brain: AIBrain;
  systemPrompt?: string;
  maxTokens?: number;
  expectJSON?: boolean;
  timeout?: number;
  logger: Logger;
}

export interface AIUsage {
  inputTokens: number;
  outputTokens: number;
  estimatedCostUSD: number;
}

// ── Cost constants (per 1M tokens) ──────────────

const COST_PER_1M: Record<AIBrain, { input: number; output: number }> = {
  claude_sonnet_4_6: { input: 3.0, output: 15.0 },
  gpt_5_4: { input: 2.5, output: 10.0 },
};

function estimateCost(brain: AIBrain, inputTokens: number, outputTokens: number): number {
  const rates = COST_PER_1M[brain];
  return (inputTokens / 1_000_000) * rates.input + (outputTokens / 1_000_000) * rates.output;
}

// ── JSON extraction helper ──────────────────────

function extractJSON(raw: string): string {
  // Strip markdown fences if present
  const fenceMatch = raw.match(/```(?:json)?\s*\n?([\s\S]*?)```/);
  if (fenceMatch) {
    return fenceMatch[1]!.trim();
  }
  // Find outermost JSON object: first { to last }
  const firstBrace = raw.indexOf('{');
  const lastBrace = raw.lastIndexOf('}');
  if (firstBrace !== -1 && lastBrace > firstBrace) {
    return raw.slice(firstBrace, lastBrace + 1);
  }
  // Try array: first [ to last ]
  const firstBracket = raw.indexOf('[');
  const lastBracket = raw.lastIndexOf(']');
  if (firstBracket !== -1 && lastBracket > firstBracket) {
    return raw.slice(firstBracket, lastBracket + 1);
  }
  return raw.trim();
}

// ── Claude implementation ───────────────────────

async function askClaude(
  userPrompt: string,
  options: AIClientOptions,
): Promise<{ text: string; usage: AIUsage }> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    throw new Error('ANTHROPIC_API_KEY is not set. Please add it to your .env file.');
  }

  const client = new Anthropic({ apiKey });
  const maxTokens = options.maxTokens ?? 4096;
  const timeoutMs = options.timeout ?? 60000;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const message = await client.messages.create(
      {
        model: 'claude-sonnet-4-6',
        max_tokens: maxTokens,
        ...(options.systemPrompt ? { system: options.systemPrompt } : {}),
        messages: [{ role: 'user', content: userPrompt }],
      },
      { signal: controller.signal },
    );

    const textBlock = message.content.find((b) => b.type === 'text');
    const text = textBlock && textBlock.type === 'text' ? textBlock.text : '';
    const inputTokens = message.usage.input_tokens;
    const outputTokens = message.usage.output_tokens;

    return {
      text,
      usage: {
        inputTokens,
        outputTokens,
        estimatedCostUSD: estimateCost('claude_sonnet_4_6', inputTokens, outputTokens),
      },
    };
  } finally {
    clearTimeout(timer);
  }
}

// ── GPT implementation ──────────────────────────

async function askGPT(
  userPrompt: string,
  options: AIClientOptions,
): Promise<{ text: string; usage: AIUsage }> {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error('OPENAI_API_KEY is not set. Please add it to your .env file.');
  }

  const client = new OpenAI({ apiKey });
  const maxTokens = options.maxTokens ?? 4096;
  const timeoutMs = options.timeout ?? 60000;

  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [];
  if (options.systemPrompt) {
    messages.push({ role: 'system', content: options.systemPrompt });
  }
  messages.push({ role: 'user', content: userPrompt });

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await client.chat.completions.create(
      {
        model: 'gpt-5.4',
        max_tokens: maxTokens,
        messages,
      },
      { signal: controller.signal },
    );

    const text = response.choices[0]?.message?.content ?? '';
    const inputTokens = response.usage?.prompt_tokens ?? 0;
    const outputTokens = response.usage?.completion_tokens ?? 0;

    return {
      text,
      usage: {
        inputTokens,
        outputTokens,
        estimatedCostUSD: estimateCost('gpt_5_4', inputTokens, outputTokens),
      },
    };
  } finally {
    clearTimeout(timer);
  }
}

// ── JSON repair helper ─────────────────────────

function repairJSON(raw: string): string {
  let s = raw;

  // Remove trailing commas before } or ]
  s = s.replace(/,\s*([}\]])/g, '$1');

  // Remove double commas
  s = s.replace(/,\s*,/g, ',');

  // Remove characters after the last } or ]
  const lastBrace = s.lastIndexOf('}');
  const lastBracket = s.lastIndexOf(']');
  const lastClose = Math.max(lastBrace, lastBracket);
  if (lastClose !== -1 && lastClose < s.length - 1) {
    s = s.slice(0, lastClose + 1);
  }

  // Remove characters before the first { or [
  const firstBrace = s.indexOf('{');
  const firstBracket = s.indexOf('[');
  const firstOpen = firstBrace !== -1 && firstBracket !== -1
    ? Math.min(firstBrace, firstBracket)
    : firstBrace !== -1 ? firstBrace : firstBracket;
  if (firstOpen > 0) {
    s = s.slice(firstOpen);
  }

  return s;
}

// ── Public API ──────────────────────────────────

export async function askAI(
  userPrompt: string,
  options: AIClientOptions,
): Promise<{ text: string; usage: AIUsage }> {
  const { brain, logger } = options;
  const startTime = Date.now();

  logger.info('AI request started', { brain });

  const result = brain === 'claude_sonnet_4_6'
    ? await askClaude(userPrompt, options)
    : await askGPT(userPrompt, options);

  const durationMs = Date.now() - startTime;

  logger.info('AI request completed', {
    brain,
    inputTokens: result.usage.inputTokens,
    outputTokens: result.usage.outputTokens,
    estimatedCostUSD: result.usage.estimatedCostUSD.toFixed(4),
    durationMs,
  });

  return result;
}

export async function askAIJSON<T = unknown>(
  userPrompt: string,
  options: AIClientOptions,
): Promise<{ data: T; usage: AIUsage }> {
  const result = await askAI(userPrompt, { ...options, expectJSON: true });
  const cleaned = extractJSON(result.text);

  // Try parsing as-is first
  try {
    const data = JSON.parse(cleaned) as T;
    return { data, usage: result.usage };
  } catch {
    options.logger.warn('JSON parse failed, attempting repair', {
      rawLength: result.text.length,
    });
  }

  // Try with repair
  const repaired = repairJSON(cleaned);
  try {
    const data = JSON.parse(repaired) as T;
    options.logger.info('JSON repair succeeded');
    return { data, usage: result.usage };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    options.logger.error('JSON repair also failed', { error: message });
    throw new Error(`AI returned invalid JSON: ${message}`);
  }
}
