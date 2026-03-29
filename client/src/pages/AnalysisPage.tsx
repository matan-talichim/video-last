import { useState, useEffect, useRef, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useParams, useNavigate } from 'react-router-dom';

type SubStepStatus = 'pending' | 'processing' | 'done' | 'error';

interface SubStepProgress {
  status: SubStepStatus;
  durationMs?: number;
  error?: string;
}

interface JobStatus {
  jobId: string;
  status: 'idle' | 'processing' | 'preprocessed' | 'presenter_detected' | 'error';
  currentStep?: string;
  progress?: {
    audio: SubStepProgress;
    proxy: SubStepProgress;
    frames: SubStepProgress;
    presenterDetection: SubStepProgress;
  };
}

const POLL_INTERVAL = 2000;

function AnalysisPage() {
  const { t } = useTranslation();
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();

  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [startError, setStartError] = useState(false);
  const startedRef = useRef(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const pollStatus = useCallback(async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}/status`);
      if (res.ok) {
        const data = (await res.json()) as JobStatus;
        setJobStatus(data);

        if (data.status === 'presenter_detected' || data.status === 'error') {
          if (pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = null;
          }
        }
      }
    } catch {
      // Silently retry on next poll
    }
  }, [jobId]);

  useEffect(() => {
    if (!jobId || startedRef.current) return;
    startedRef.current = true;

    const startProcessing = async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}/process`, { method: 'POST' });
        if (!res.ok) {
          setStartError(true);
          return;
        }

        // Start polling
        pollRef.current = setInterval(pollStatus, POLL_INTERVAL);
        // Also poll immediately
        pollStatus();
      } catch {
        setStartError(true);
      }
    };

    startProcessing();

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [jobId, pollStatus]);

  const progress = jobStatus?.progress ?? {
    audio: { status: 'pending' as SubStepStatus },
    proxy: { status: 'pending' as SubStepStatus },
    frames: { status: 'pending' as SubStepStatus },
    presenterDetection: { status: 'pending' as SubStepStatus },
  };

  const isDone = jobStatus?.status === 'presenter_detected';

  return (
    <div className="flex flex-col items-center gap-8">
      <h2 className="text-3xl font-bold text-white">{t('analysis.title')}</h2>

      <div className="w-full max-w-lg rounded-xl border border-gray-800 bg-gray-900 px-8 py-8">
        {startError ? (
          <p className="text-center text-red-400">{t('analysis.startError')}</p>
        ) : (
          <>
            {!isDone && (
              <p className="mb-6 text-center text-lg text-gray-300">{t('analysis.processing')}</p>
            )}

            <div className="flex flex-col gap-4">
              <StepRow
                label={t('analysis.extractAudio')}
                step={progress.audio}
              />
              <StepRow
                label={t('analysis.createProxy')}
                step={progress.proxy}
              />
              <StepRow
                label={t('analysis.extractFrames')}
                step={progress.frames}
              />
              <StepRow
                label={t('analysis.presenterDetection')}
                step={progress.presenterDetection}
              />
            </div>

            {isDone && (
              <p className="mt-6 text-center text-lg font-semibold text-green-400">
                {t('analysis.allDone')}
              </p>
            )}
          </>
        )}
      </div>

      <button
        onClick={() => navigate(`/settings/${jobId}`)}
        className="rounded-xl border border-gray-700 bg-gray-800 px-6 py-2.5 text-gray-300 transition-colors hover:border-gray-500 hover:text-white"
      >
        {t('analysis.backToSettings')}
      </button>
    </div>
  );
}

function StepRow({ label, step }: { label: string; step?: SubStepProgress }) {
  const { t } = useTranslation();
  const safeStep = step ?? { status: 'pending' as SubStepStatus };

  return (
    <div className="flex items-center justify-between rounded-lg border border-gray-700 bg-gray-800 px-4 py-3">
      <div className="flex items-center gap-3">
        <StatusIcon status={safeStep.status} />
        <span className="font-medium text-gray-200">{label}</span>
      </div>
      <span className="text-sm text-gray-400">
        <StatusLabel status={safeStep.status} durationMs={safeStep.durationMs} t={t} />
      </span>
    </div>
  );
}

function StatusIcon({ status }: { status: SubStepStatus }) {
  switch (status) {
    case 'done':
      return <span className="text-xl text-green-400">✓</span>;
    case 'processing':
      return (
        <div className="h-5 w-5 animate-spin rounded-full border-2 border-gray-600 border-t-blue-400" />
      );
    case 'error':
      return <span className="text-xl text-red-400">✗</span>;
    default:
      return <span className="h-5 w-5 rounded-full border-2 border-gray-600" />;
  }
}

function StatusLabel({
  status,
  durationMs,
  t,
}: {
  status: SubStepStatus;
  durationMs?: number;
  t: (key: string, options?: Record<string, unknown>) => string;
}) {
  switch (status) {
    case 'done':
      return durationMs != null
        ? t('analysis.completedTime', { seconds: (durationMs / 1000).toFixed(1) })
        : t('analysis.completed');
    case 'processing':
      return t('analysis.inProgress');
    case 'error':
      return t('analysis.error');
    default:
      return t('analysis.pending');
  }
}

export default AnalysisPage;
