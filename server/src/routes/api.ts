import { Router } from 'express';
import type { Request, Response, NextFunction } from 'express';
import multer from 'multer';
import { randomUUID } from 'node:crypto';
import { mkdirSync, existsSync, writeFileSync, readFileSync, readdirSync, statSync, createReadStream } from 'node:fs';
import { resolve, extname } from 'node:path';
import { rename } from 'node:fs/promises';
import type { AppConfig } from '../utils/config.js';
import { getMediaInfo } from '../utils/ffmpeg.js';
import { createLogger } from '../utils/logger.js';
import { runPreprocess } from '../pipeline/steps/preprocess.js';
import { runPresenterDetection } from '../pipeline/steps/presenter-detection.js';
import { runTranscription } from '../pipeline/steps/transcription.js';
import { runMergeAndClean } from '../pipeline/steps/merge-and-clean.js';
import { runAnalyze } from '../pipeline/steps/analyze.js';
import { runEditAssembly } from '../pipeline/steps/edit-assembly.js';
import { askAIJSON } from '../utils/ai-client.js';
import type { AIBrain } from '../utils/ai-client.js';

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

      const jobDir = resolve(config.editor.inputDir, String(jobId));
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
      const jobDir = resolve(config.editor.inputDir, String(jobId));

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
      const jobDir = resolve(config.editor.inputDir, String(jobId));

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

      // Guard: reject if already processing or further along in the pipeline
      const statusPath = resolve(jobDir, 'status.json');
      if (existsSync(statusPath)) {
        const current = JSON.parse(readFileSync(statusPath, 'utf-8')) as { status: string };
        const blockedStatuses = ['processing', 'analyzed', 'editing', 'edited'];
        if (blockedStatuses.includes(current.status)) {
          logger.warn('Process — blocked, current status prevents re-run', { jobId, currentStatus: current.status });
          res.status(409).json({ error: `Job is already in status: ${current.status}` });
          return;
        }
      }

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

      // Get presenter detection config
      const pdConfig = (config as unknown as Record<string, unknown>).presenterDetection as
        { lipThreshold?: number; vadThreshold?: number; buffer?: number; pythonPath?: string } | undefined;
      const presenterConfig = {
        lipThreshold: pdConfig?.lipThreshold ?? 0.15,
        vadThreshold: pdConfig?.vadThreshold ?? 0.5,
        buffer: pdConfig?.buffer ?? 0.25,
        pythonPath: pdConfig?.pythonPath ?? 'python3',
      };

      // Get transcription config
      const txConfig = (config as unknown as Record<string, unknown>).transcription as
        { model?: string; language?: string; smartFormat?: boolean; utterances?: boolean; punctuate?: boolean; words?: boolean; paragraphs?: boolean } | undefined;
      const transcriptionConfig = {
        model: txConfig?.model ?? 'nova-3',
        language: txConfig?.language ?? 'he',
        smartFormat: txConfig?.smartFormat ?? true,
        utterances: txConfig?.utterances ?? true,
        punctuate: txConfig?.punctuate ?? true,
        words: txConfig?.words ?? true,
        paragraphs: txConfig?.paragraphs ?? true,
      };

      // Read settings for AI brain selection
      const settings = JSON.parse(readFileSync(settingsPath, 'utf-8')) as { aiBrain?: string };
      const aiBrain: AIBrain = settings.aiBrain === 'gpt_5_4' ? 'gpt_5_4' : 'claude_sonnet_4_6';

      // Run preprocess → presenter detection → transcription → merge-and-clean in the background (do not await)
      runPreprocess(jobDir, inputFile, config.ffmpeg, logger, options)
        .then(() => {
          logger.info('Preprocess done, starting presenter detection', { jobId });
          return runPresenterDetection(jobDir, presenterConfig, logger);
        })
        .then(() => {
          logger.info('Presenter detection done, starting transcription', { jobId });
          return runTranscription(jobDir, transcriptionConfig, logger);
        })
        .then(() => {
          logger.info('Transcription done, starting merge and clean', { jobId });
          return runMergeAndClean(jobDir, aiBrain, logger);
        })
        .then(() => {
          logger.info('Merge and clean done, starting analysis', { jobId });
          return runAnalyze(jobDir, aiBrain, logger);
        })
        .catch((err) => {
          const errorMsg = err instanceof Error ? err.message : String(err);
          logger.error('Process failed', { jobId, error: errorMsg });
          // Read current status to preserve progress info
          const currentStatus = existsSync(statusPath)
            ? JSON.parse(readFileSync(statusPath, 'utf-8'))
            : {};
          const currentProgress = (currentStatus as Record<string, unknown>).progress ?? {};
          writeFileSync(
            statusPath,
            JSON.stringify({
              ...currentStatus,
              status: 'error',
              error: errorMsg,
              progress: {
                ...(currentProgress as Record<string, unknown>),
              },
            }, null, 2),
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
      const jobDir = resolve(config.editor.inputDir, String(jobId));

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
            presenterDetection: { status: 'pending' },
            transcription: { status: 'pending' },
            mergeAndClean: { status: 'pending' },
            analysis: { status: 'pending' },
            editAssembly: { status: 'pending' },
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
  // GET /api/jobs/:jobId/analysis
  // ──────────────────────────────────────────
  router.get('/jobs/:jobId/analysis', (req, res) => {
    try {
      const { jobId } = req.params;
      const jobDir = resolve(config.editor.inputDir, String(jobId));

      if (!existsSync(jobDir)) {
        res.status(404).json({ error: 'Job not found' });
        return;
      }

      const analysisPath = resolve(jobDir, 'analysis.json');
      if (!existsSync(analysisPath)) {
        res.status(404).json({ error: 'Analysis not ready yet' });
        return;
      }

      const analysisData = JSON.parse(readFileSync(analysisPath, 'utf-8'));
      res.json(analysisData);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      logger.error('Analysis fetch failed', { error: message });
      res.status(500).json({ error: message });
    }
  });

  // ──────────────────────────────────────────
  // POST /api/jobs/:jobId/analysis/approve
  // ──────────────────────────────────────────
  router.post('/jobs/:jobId/analysis/approve', (req, res) => {
    try {
      const { jobId } = req.params;
      const jobDir = resolve(config.editor.inputDir, String(jobId));

      if (!existsSync(jobDir)) {
        logger.warn('Approve — job not found', { jobId });
        res.status(404).json({ error: 'Job not found' });
        return;
      }

      const plan = req.body as Record<string, unknown>;
      const selectedHook = typeof plan.selectedHook === 'number' ? plan.selectedHook : 0;
      const approvedPlan = { ...plan, selectedHook };
      const approvedPath = resolve(jobDir, 'approved_plan.json');
      writeFileSync(approvedPath, JSON.stringify(approvedPlan, null, 2), 'utf-8');

      logger.info('Selected hook saved', { jobId, selectedHook });

      // Update status
      const statusPath = resolve(jobDir, 'status.json');
      if (existsSync(statusPath)) {
        const currentStatus = JSON.parse(readFileSync(statusPath, 'utf-8')) as Record<string, unknown>;
        const currentProgress = (currentStatus.progress ?? {}) as Record<string, unknown>;
        writeFileSync(statusPath, JSON.stringify({
          ...currentStatus,
          status: 'editing',
          progress: {
            ...currentProgress,
            editAssembly: { status: 'processing' },
          },
        }, null, 2), 'utf-8');
      }

      logger.info('Plan approved, starting edit assembly', { jobId });

      // Auto-trigger edit assembly in the background
      runEditAssembly(jobDir, config.ffmpeg, logger).catch((err) => {
        const errorMsg = err instanceof Error ? err.message : String(err);
        logger.error('Edit assembly failed after approve', { jobId, error: errorMsg });
      });

      res.json({ status: 'editing' });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      logger.error('Plan approval failed', { error: message });
      res.status(500).json({ error: message });
    }
  });

  // ──────────────────────────────────────────
  // POST /api/jobs/:jobId/analysis/revise
  // ──────────────────────────────────────────
  router.post('/jobs/:jobId/analysis/revise', async (req: Request, res: Response) => {
    const startTime = Date.now();
    const { jobId } = req.params;

    try {
      const jobDir = resolve(config.editor.inputDir, String(jobId));

      if (!existsSync(jobDir)) {
        logger.warn('Revise — job not found', { jobId });
        res.status(404).json({ error: 'Job not found' });
        return;
      }

      const analysisPath = resolve(jobDir, 'analysis.json');
      if (!existsSync(analysisPath)) {
        logger.warn('Revise — analysis not found', { jobId });
        res.status(404).json({ error: 'Analysis not found' });
        return;
      }

      const { notes } = req.body as { notes?: string };
      if (!notes || notes.trim().length === 0) {
        logger.warn('Revise — empty notes', { jobId });
        res.status(400).json({ error: 'Notes are required' });
        return;
      }

      // Read current analysis and settings
      const currentAnalysis = readFileSync(analysisPath, 'utf-8');
      const settingsPath = resolve(jobDir, 'settings.json');
      const settings = existsSync(settingsPath)
        ? (JSON.parse(readFileSync(settingsPath, 'utf-8')) as { aiBrain?: string })
        : {};
      const aiBrain: AIBrain = settings.aiBrain === 'gpt_5_4' ? 'gpt_5_4' : 'claude_sonnet_4_6';

      logger.info('Revision request', { jobId, brain: aiBrain, notesLength: notes.length });

      const revisionPrompt = `קיבלת תוכנית עריכה קודמת והמשתמש ביקש שינויים.

תוכנית נוכחית:
${currentAnalysis}

הערות המשתמש:
${notes}

עדכן את התוכנית לפי ההערות. החזר JSON מלא באותו פורמט בדיוק, עם השינויים שהמשתמש ביקש.
אל תשנה דברים שהמשתמש לא ביקש לשנות.

חשוב מאוד: החזר JSON תקין בלבד. וודא שאין פסיקים מיותרים לפני } או ]. וודא שכל string סגור עם גרשיים. אל תוסיף טקסט לפני או אחרי ה-JSON.`;

      const { data, usage } = await askAIJSON(revisionPrompt, {
        brain: aiBrain,
        maxTokens: config.ai.maxTokens ?? 4096,
        timeout: config.analysis.timeout ?? 120000,
        logger,
      });

      // Save revised analysis
      writeFileSync(analysisPath, JSON.stringify(data, null, 2), 'utf-8');

      const elapsed = Date.now() - startTime;
      logger.info('Revision completed', {
        jobId,
        brain: aiBrain,
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
        costUSD: usage.estimatedCostUSD,
        elapsed: `${elapsed}ms`,
      });

      res.json(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      logger.error('Revision failed', { jobId, error: message });

      const isJsonError = message.includes('invalid JSON') || message.includes('JSON at position');
      res.status(isJsonError ? 502 : 500).json({
        error: isJsonError
          ? 'המוח החזיר תשובה לא תקינה, נסה שוב'
          : message,
      });
    }
  });

  // ──────────────────────────────────────────
  // POST /api/jobs/:jobId/edit
  // ──────────────────────────────────────────
  router.post('/jobs/:jobId/edit', (req, res) => {
    try {
      const { jobId } = req.params;
      const jobDir = resolve(config.editor.inputDir, String(jobId));

      if (!existsSync(jobDir)) {
        res.status(404).json({ error: 'Job not found' });
        return;
      }

      const approvedPlanPath = resolve(jobDir, 'approved_plan.json');
      if (!existsSync(approvedPlanPath)) {
        res.status(400).json({ error: 'יש לאשר תוכנית עריכה לפני ביצוע' });
        return;
      }

      // Guard: reject if already editing
      const statusPath = resolve(jobDir, 'status.json');
      if (existsSync(statusPath)) {
        const current = JSON.parse(readFileSync(statusPath, 'utf-8')) as { status: string };
        if (current.status === 'editing') {
          res.status(409).json({ error: 'Edit already in progress' });
          return;
        }
      }

      logger.info('Edit triggered manually', { jobId });

      runEditAssembly(jobDir, config.ffmpeg, logger).catch((err) => {
        const errorMsg = err instanceof Error ? err.message : String(err);
        logger.error('Edit assembly failed', { jobId, error: errorMsg });
      });

      res.json({ status: 'editing' });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      logger.error('Edit endpoint failed', { error: message });
      res.status(500).json({ error: message });
    }
  });

  // ──────────────────────────────────────────
  // GET /api/jobs/:jobId/result
  // ──────────────────────────────────────────
  router.get('/jobs/:jobId/result', (req, res) => {
    try {
      const { jobId } = req.params;
      const jobDir = resolve(config.editor.inputDir, String(jobId));

      if (!existsSync(jobDir)) {
        res.status(404).json({ error: 'Job not found' });
        return;
      }

      const statusPath = resolve(jobDir, 'status.json');
      if (!existsSync(statusPath)) {
        res.json({ status: 'pending' });
        return;
      }

      const statusData = JSON.parse(readFileSync(statusPath, 'utf-8')) as Record<string, unknown>;
      const status = statusData.status as string;

      if (status === 'editing') {
        res.json({ status: 'editing' });
        return;
      }

      if (status === 'edited') {
        res.json({
          status: 'edited',
          editResult: statusData.editResult,
        });
        return;
      }

      if (status === 'error') {
        res.json({
          status: 'error',
          error: statusData.error,
        });
        return;
      }

      res.json({ status });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      logger.error('Result fetch failed', { error: message });
      res.status(500).json({ error: message });
    }
  });

  // ──────────────────────────────────────────
  // GET /api/jobs/:jobId/video/edited
  // ──────────────────────────────────────────
  router.get('/jobs/:jobId/video/edited', (req, res) => {
    try {
      const { jobId } = req.params;
      const jobDir = resolve(config.editor.inputDir, String(jobId));

      if (!existsSync(jobDir)) {
        res.status(404).json({ error: 'Job not found' });
        return;
      }

      const videoPath = resolve(jobDir, 'edited.mp4');
      if (!existsSync(videoPath)) {
        res.status(404).json({ error: 'Edited video not ready' });
        return;
      }

      const fileStat = statSync(videoPath);
      const fileSize = fileStat.size;
      const range = req.headers.range;

      if (range) {
        const parts = range.replace(/bytes=/, '').split('-');
        const start = parseInt(parts[0]!, 10);
        const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
        const chunkSize = end - start + 1;

        const stream = createReadStream(videoPath, { start, end });

        res.writeHead(206, {
          'Content-Range': `bytes ${start}-${end}/${fileSize}`,
          'Accept-Ranges': 'bytes',
          'Content-Length': chunkSize,
          'Content-Type': 'video/mp4',
        });

        stream.pipe(res);
      } else {
        res.writeHead(200, {
          'Content-Length': fileSize,
          'Content-Type': 'video/mp4',
          'Accept-Ranges': 'bytes',
        });

        const stream = createReadStream(videoPath);
        stream.pipe(res);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      logger.error('Video stream failed', { error: message });
      res.status(500).json({ error: message });
    }
  });

  return router;
}
