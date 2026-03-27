import { writeFileSync, appendFileSync, mkdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import chalk from 'chalk';

export interface LoggerOptions {
  logsDir: string;
  level: 'debug' | 'info' | 'warn' | 'error';
  sessionId?: string;
}

export interface Logger {
  debug: (message: string, data?: unknown) => void;
  info: (message: string, data?: unknown) => void;
  warn: (message: string, data?: unknown) => void;
  error: (message: string, data?: unknown) => void;
}

const LOG_LEVELS: Record<string, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

export function createLogger(options: LoggerOptions): Logger {
  const { logsDir, level, sessionId } = options;
  const minLevel = LOG_LEVELS[level] ?? 1;
  const logFileName = sessionId
    ? `session-${sessionId}.log`
    : `app-${new Date().toISOString().split('T')[0]}.log`;
  const logFilePath = join(logsDir, logFileName);

  if (!existsSync(logsDir)) {
    mkdirSync(logsDir, { recursive: true });
  }

  function formatTimestamp(): string {
    return new Date().toISOString();
  }

  function writeToFile(levelName: string, message: string, data?: unknown): void {
    const line = data
      ? `[${formatTimestamp()}] [${levelName.toUpperCase()}] ${message} ${JSON.stringify(data)}\n`
      : `[${formatTimestamp()}] [${levelName.toUpperCase()}] ${message}\n`;
    appendFileSync(logFilePath, line, 'utf-8');
  }

  function log(levelName: string, colorFn: (s: string) => string, message: string, data?: unknown): void {
    if (LOG_LEVELS[levelName]! < minLevel) return;

    const timestamp = formatTimestamp();
    const prefix = colorFn(`[${timestamp}] [${levelName.toUpperCase()}]`);
    if (data) {
      process.stderr.write(`${prefix} ${message} ${JSON.stringify(data)}\n`);
    } else {
      process.stderr.write(`${prefix} ${message}\n`);
    }

    writeToFile(levelName, message, data);
  }

  return {
    debug: (message: string, data?: unknown) => log('debug', chalk.gray, message, data),
    info: (message: string, data?: unknown) => log('info', chalk.blue, message, data),
    warn: (message: string, data?: unknown) => log('warn', chalk.yellow, message, data),
    error: (message: string, data?: unknown) => log('error', chalk.red, message, data),
  };
}
