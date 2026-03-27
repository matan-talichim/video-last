import { Router } from 'express';
import type { Request, Response, NextFunction } from 'express';
import multer from 'multer';
import { randomUUID } from 'node:crypto';
import { mkdirSync, existsSync, writeFileSync, readFileSync, readdirSync } from 'node:fs';
import { resolve, extname } from 'node:path';
import { rename } from 'node:fs/promises';
import type { AppConfig } from '../utils/config.js';
import { getMediaInfo } from '../utils/ffmpeg.js';
import { createLogger } from '../utils/logger.js';
import { runPreprocess } from '../pipeline/steps/preprocess.js';

export function createApiRouter(config: AppConfig): Router {
  const router = Router();
  const logger = createLogger({
    logsDir: config.editor.logsDir,
    level: config.editor.logLevel,
  });

  // Multer setup — store in temp, then move to job folder
  const upload = multer({
    dest: resolve(config.editor.inputDir, '_uploads_tmp'),
    limits: { fileSize: config.upload.maxFileSize },
    fileFilter: (_req, file, cb) => {
      const ext = extname(file.originalname).toLowerCase().replace('.', '');
      if (config.upload.allowedFormats.includes(ext)) {
        cb(null, true);
      } else {
        cb(new Error(`Unsupported format: .${ext}`));
      }
    },
  });

  // ──────────────────────────────────────────
  // GET /api/health
  // ──────────────────────────────────────────
  router.get('/health', (_req, res) => {
    res.json({ status: 'ok', version: '0.1.0' });
  });

  // ──────────────────────────────────────────
  // GET /api/config
  // ──────────────────────────────────────────
  router.get('/config', (_req, res) => {
    const safeConfig = {
      editor: {
        inputDir: config.editor.inputDir,
        outputDir: config.editor.outputDir,
        logLevel: config.editor.logLevel,
      },
      pipeline: config.pipeline,
    };
    res.json(safeConfig);
  });

  // ──────────────────────────────────────────
  // POST /api/upload
  // ──────────────────────────────────────────
  router.post('/upload', (req: Request, res: Response, next: NextFunction) => {
    upload.single('video')(req, res, (err: unknown) => {
      if (err) {
        const message = err instanceof Error ? err.message : String(err);
        logger.warn('Upload rejected', { error: message });
        res.status(400).json({ error: message });
        return;
      }
      next();
    });
  }, async (req: Request, res: Response) => {
    const startTime = Date.now();

    try {
      if (!req.file) {
        logger.warn('Upload attempt with no file');
        res.status(400).json({ error: 'No video file provided' });
        return;
      }

      const jobId = randomUUID();
      const ext = extname(req.file.originalname).toLowerCase();
      const jobDir = resolve(config.editor.inputDir, jobId);
      const destPath = resolve(jobDir, `original${ext}`);

      logger.info('Upload started', {
        jobId,
        originalName: req.file.originalname,
        size: req.file.size,
      });

      // Create job directory and move file
      mkdirSync(jobDir, { recursive: true });
      await rename(req.file.path, destPath);

      // Extract media info via ffprobe
      let mediaInfo = { duration: 0, width: 0, height: 0, fps: 0, codec: 'unknown', audioCodec: 'unknown', fileSize: 0, filePath: destPath };
      try {
        mediaInfo = await getMediaInfo(destPath, config.ffmpeg, logger);
      } catch (probeErr) {
        logger.warn('FFprobe failed — returning file without media info', {
          error: probeErr instanceof Error ? probeErr.message : String(probeErr),
        });
      }

      const result = {
        jobId,
        fileName: req.file.originalname,
        fileSize: req.file.size,
        duration: mediaInfo.duration,
        width: mediaInfo.width,
        height: mediaInfo.height,
        fps: Math.round(mediaInfo.fps * 100) / 100,
        codec: mediaInfo.codec,
        audioCodec: mediaInfo.audioCodec,
        status: 'uploaded',
      };

      const elapsed = Date.now() - startTime;
      logger.info('Upload completed', { jobId, elapsed: `${elapsed}ms` });

      res.json(result);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      logger.error('Upload failed', { error: message });
      res.status(500).json({ error: message });
    }
  });

  // ──────────────────────────────────────────
  // POST /api/jobs/:jobId/settings
  // ──────────────────────────────────────────
  router.post('/jobs/:jobId/settings', (req, res) => {
    try {
      const { jobId } = req.params;
      const settings = req.body;

      const jobDir = resolve(config.editor.inputDir, jobId!);
      if (!existsSync(jobDir)) {
        logger.warn('Settings save — job not found', { jobId });
        res.status(404).json({ error: 'Job not found' });
        return;
      }

      const settingsPath = resolve(jobDir, 'settings.json');
      writeFileSync(settingsPath, JSON.stringify(settings, null, 2), 'utf-8');

      logger.info('Settings saved', { jobId, settings });

      res.json({
        jobId,
        status: 'configured',
        settings,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      logger.error('Settings save failed', { error: message });
      res.status(500).json({ error: message });
    }
  });

  // ──────────────────────────────────────────
  // GET /api/jobs/:jobId
  // ──────────────────────────────────────────
  router.get('/jobs/:jobId', (req, res) => {
    try {
      const { jobId } = req.params;
      const jobDir = resolve(config.editor.inputDir, jobId!);

      if (!existsSync(jobDir)) {
        res.status(404).json({ error: 'Job not found' });
        return;
      }

      const settingsPath = resolve(jobDir, 'settings.json');
      const hasSettings = existsSync(settingsPath);
      const settings = hasSettings
        ? JSON.parse(readFileSync(settingsPath, 'utf-8'))
        : null;

      res.json({
        jobId,
        status: hasSettings ? 'configured' : 'uploaded',
        settings,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      logger.error('Job fetch failed', { error: message });
      res.status(500).json({ error: message });
    }
  });

  // ──────────────────────────────────────────
  // POST /api/jobs/:jobId/process
  // ──────────────────────────────────────────
  router.post('/jobs/:jobId/process', (req, res) => {
    try {
      const { jobId } = req.params;
      const jobDir = resolve(config.editor.inputDir, jobId!);

      if (!existsSync(jobDir)) {
        logger.warn('Process — job not found', { jobId });
        res.status(404).json({ error: 'Job not found' });
        return;
      }

      const settingsPath = resolve(jobDir, 'settings.json');
      if (!existsSync(settingsPath)) {
        logger.warn('Process — settings not found', { jobId });
        res.status(400).json({ error: 'Settings not configured' });
        return;
      }

      // Find the original video file
      const files = readdirSync(jobDir).filter((f) => f.startsWith('original.'));
      if (files.length === 0) {
        logger.warn('Process — original file not found', { jobId });
        res.status(400).json({ error: 'Original video file not found' });
        return;
      }

      const inputFile = resolve(jobDir, files[0]!);

      // Update status to processing
      const statusPath = resolve(jobDir, 'status.json');
      writeFileSync(statusPath, JSON.stringify({ status: 'processing' }, null, 2), 'utf-8');

      logger.info('Process started', { jobId, inputFile });

      // Get preprocess options from config
      const preprocessStep = config.pipeline.steps.find((s) => s.name === 'preprocess');
      const stepOptions = preprocessStep?.options ?? {};
      const options = {
        audioSampleRate: (stepOptions.audioSampleRate as number) ?? 16000,
        proxyHeight: (stepOptions.proxyHeight as number) ?? 480,
        proxyFps: (stepOptions.proxyFps as number) ?? 10,
        frameInterval: (stepOptions.frameInterval as number) ?? 5,
      };

      // Run preprocess in the background (do not await)
      runPreprocess(jobDir, inputFile, config.ffmpeg, logger, options).catch((err) => {
        const errorMsg = err instanceof Error ? err.message : String(err);
        logger.error('Process failed', { jobId, error: errorMsg });
        writeFileSync(
          statusPath,
          JSON.stringify({ status: 'error', error: errorMsg }, null, 2),
          'utf-8',
        );
      });

      res.json({ jobId, status: 'processing' });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      logger.error('Process endpoint failed', { error: message });
      res.status(500).json({ error: message });
    }
  });

  // ──────────────────────────────────────────
  // GET /api/jobs/:jobId/status
  // ──────────────────────────────────────────
  router.get('/jobs/:jobId/status', (req, res) => {
    try {
      const { jobId } = req.params;
      const jobDir = resolve(config.editor.inputDir, jobId!);

      if (!existsSync(jobDir)) {
        res.status(404).json({ error: 'Job not found' });
        return;
      }

      const statusPath = resolve(jobDir, 'status.json');
      if (!existsSync(statusPath)) {
        res.json({
          jobId,
          status: 'idle',
          currentStep: 'none',
          progress: {
            audio: { status: 'pending' },
            proxy: { status: 'pending' },
            frames: { status: 'pending' },
          },
        });
        return;
      }

      const statusData = JSON.parse(readFileSync(statusPath, 'utf-8'));
      res.json({ jobId, ...statusData });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      logger.error('Status fetch failed', { error: message });
      res.status(500).json({ error: message });
    }
  });

  // ──────────────────────────────────────────
  // POST /api/edit (placeholder)
  // ──────────────────────────────────────────
  router.post('/edit', (_req, res) => {
    res.json({ message: 'not implemented yet' });
  });

  return router;
}
