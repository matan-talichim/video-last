import { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useParams, useNavigate } from 'react-router-dom';

// ── Types ────────────────────────────────────────

interface EditResult {
  outputPath: string;
  originalDuration: number;
  editedDuration: number;
  segmentsKept: number;
  segmentsRemoved: number;
  compressionRatio: string;
}

interface ResultData {
  status: string;
  editResult?: EditResult;
  error?: string;
}

// ── Helpers ──────────────────────────────────────

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

// ── Main Component ───────────────────────────────

function ResultPage() {
  const { t } = useTranslation();
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();

  const [status, setStatus] = useState<string>('editing');
  const [editResult, setEditResult] = useState<EditResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retrying, setRetrying] = useState(false);

  const fetchResult = useCallback(async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}/result`);
      if (!res.ok) {
        setError('fetchError');
        return;
      }
      const data = (await res.json()) as ResultData;
      setStatus(data.status);

      if (data.status === 'edited' && data.editResult) {
        setEditResult(data.editResult);
      }

      if (data.status === 'error') {
        setError(data.error ?? 'unknown error');
      }
    } catch {
      setError('fetchError');
    }
  }, [jobId]);

  // Poll while editing
  useEffect(() => {
    fetchResult();

    const interval = setInterval(() => {
      if (status === 'editing') {
        fetchResult();
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [fetchResult, status]);

  const handleRetry = async () => {
    setRetrying(true);
    setError(null);
    setStatus('editing');

    try {
      const res = await fetch(`/api/jobs/${jobId}/edit`, { method: 'POST' });
      if (!res.ok) {
        const data = (await res.json()) as { error?: string };
        setError(data.error ?? 'retry failed');
        setStatus('error');
      }
    } catch {
      setError('network error');
      setStatus('error');
    } finally {
      setRetrying(false);
    }
  };

  // ── Error state ──
  if (error && status !== 'editing') {
    return (
      <div className="mx-auto flex max-w-xl flex-col items-center gap-6 pt-12">
        <div className="rounded-xl border border-red-800 bg-red-900/20 p-8 text-center">
          <p className="mb-2 text-xl text-red-400">{t('analysis.error')}</p>
          <p className="mb-6 text-gray-400">{error}</p>
          <div className="flex justify-center gap-4">
            <button
              onClick={handleRetry}
              disabled={retrying}
              className="rounded-xl bg-red-600 px-6 py-2.5 font-semibold text-white transition-colors hover:bg-red-500 disabled:opacity-50"
            >
              {retrying ? t('analysis.inProgress') : t('analysis.retry')}
            </button>
            <button
              onClick={() => navigate(`/preview/${jobId}`)}
              className="rounded-xl border border-gray-700 bg-gray-800 px-6 py-2.5 text-gray-300 transition-colors hover:border-gray-500 hover:text-white"
            >
              {t('result.backToPlan')}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ── Editing state ──
  if (status === 'editing' || !editResult) {
    return (
      <div className="mx-auto flex max-w-xl flex-col items-center gap-6 pt-12">
        <h2 className="text-2xl font-bold text-white">{t('result.title')}</h2>
        <div className="w-full rounded-xl border border-gray-800 bg-gray-900 p-8">
          <div className="flex flex-col items-center gap-4">
            <div className="h-10 w-10 animate-spin rounded-full border-3 border-gray-600 border-t-blue-400" />
            <p className="text-lg text-gray-300">{t('result.editing')}</p>
            <div className="h-2 w-full overflow-hidden rounded-full bg-gray-800">
              <div className="h-full animate-pulse rounded-full bg-blue-500" style={{ width: '60%' }} />
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ── Ready state ──
  const removedSeconds = editResult.originalDuration - editResult.editedDuration;
  const removedPercent = editResult.originalDuration > 0
    ? Math.round((removedSeconds / editResult.originalDuration) * 100)
    : 0;

  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-6 pb-12">
      <h2 className="text-center text-3xl font-bold text-white">{t('result.title')}</h2>

      {/* Success banner */}
      <div className="rounded-xl border border-green-800 bg-green-900/20 p-4 text-center">
        <p className="text-lg font-semibold text-green-400">{t('result.ready')}</p>
      </div>

      {/* Video player */}
      <div className="overflow-hidden rounded-xl border border-gray-800 bg-black">
        <video
          controls
          className="w-full"
          src={`/api/jobs/${jobId}/video/edited`}
        >
          <track kind="captions" />
        </video>
      </div>

      {/* Edit summary */}
      <div className="rounded-xl border border-gray-800 bg-gray-900 p-6">
        <h3 className="mb-4 text-xl font-semibold text-white">{t('result.summary')}</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="flex flex-col gap-1">
            <span className="text-sm text-gray-400">{t('result.originalDuration')}</span>
            <span className="text-lg text-gray-200">{formatDuration(editResult.originalDuration)}</span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-sm text-gray-400">{t('result.editedDuration')}</span>
            <span className="text-lg text-gray-200">{formatDuration(editResult.editedDuration)}</span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-sm text-gray-400">{t('result.removed')}</span>
            <span className="text-lg text-gray-200">
              {formatDuration(removedSeconds)} ({removedPercent}%)
            </span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-sm text-gray-400">{t('result.segmentsKept')}</span>
            <span className="text-lg text-gray-200">{editResult.segmentsKept}</span>
          </div>
        </div>
      </div>

      {/* Action buttons */}
      <div className="flex items-center justify-center gap-4">
        <button
          onClick={() => navigate(`/preview/${jobId}`)}
          className="rounded-xl border border-gray-700 bg-gray-800 px-6 py-3 text-gray-300 transition-colors hover:border-gray-500 hover:text-white"
        >
          {t('result.backToPlan')}
        </button>
        <a
          href={`/api/jobs/${jobId}/video/edited`}
          download="edited.mp4"
          className="rounded-xl bg-green-600 px-8 py-3 text-lg font-semibold text-white transition-colors hover:bg-green-500"
        >
          {t('result.download')}
        </a>
      </div>
    </div>
  );
}

export default ResultPage;
