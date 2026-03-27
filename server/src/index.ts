import express from 'express';
import cors from 'cors';
import { resolve } from 'node:path';
import { existsSync } from 'node:fs';
import { loadConfig } from './utils/config.js';
import { createLogger } from './utils/logger.js';
import { checkFFmpeg } from './utils/ffmpeg.js';
import { createApiRouter } from './routes/api.js';

// Register pipeline steps
import './pipeline/steps/passthrough.js';
import './pipeline/steps/preprocess.js';

const config = loadConfig();
const logger = createLogger({
  logsDir: config.editor.logsDir,
  level: config.editor.logLevel,
});

const app = express();
app.use(cors());
app.use(express.json());

// API routes
app.use('/api', createApiRouter(config));

// Serve frontend static files in production
const clientDistPath = resolve('..', 'client', 'dist');
if (existsSync(clientDistPath)) {
  app.use(express.static(clientDistPath));
  app.get('*', (_req, res) => {
    res.sendFile(resolve(clientDistPath, 'index.html'));
  });
}

const port = config.server.port;

app.listen(port, () => {
  logger.info(`Server running on http://localhost:${port}`);
});

// Check FFmpeg availability
checkFFmpeg(config.ffmpeg, logger).then((found) => {
  if (!found) {
    logger.warn('FFmpeg not found — video processing will not work');
  }
});
