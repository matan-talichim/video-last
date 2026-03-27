import { Router } from 'express';
import type { AppConfig } from '../utils/config.js';

export function createApiRouter(config: AppConfig): Router {
  const router = Router();

  router.get('/health', (_req, res) => {
    res.json({ status: 'ok', version: '0.1.0' });
  });

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

  router.post('/edit', (_req, res) => {
    res.json({ message: 'not implemented yet' });
  });

  return router;
}
