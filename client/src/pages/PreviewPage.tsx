import { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useParams, useNavigate } from 'react-router-dom';

// ── Types ────────────────────────────────────────

interface HookOption {
  text: string;
  start: number;
  end: number;
  score: number;
  explanation: string;
  isWeaker?: boolean;
}

interface StructureSection {
  type: string;
  start: number;
  end: number;
  description: string;
}

interface StrongPoint {
  text: string;
  start: number;
  end: number;
  reason: string;
  suggestedAction: string;
}

interface BrollSuggestion {
  triggerText: string;
  start: number;
  end: number;
  description: string;
  reason: string;
  priority: string;
}

interface Effect {
  type: string;
  start: number;
  reason: string;
}

interface CtaData {
  text: string;
  start: number;
  end: number;
  type: string;
}

interface EditingPlan {
  estimatedDuration: number;
  cuts: string;
  effects: Effect[];
  music: string;
  subtitles: string;
  cta: CtaData;
}

interface AnalysisData {
  summary: string;
  detectedGenre: string;
  targetAudience: string;
  viralityScore: number;
  retentionRisk: string;
  isWeakStart: boolean;
  hookOptions: HookOption[];
  structure: {
    recommended: string;
    sections: StructureSection[];
  };
  strongPoints: StrongPoint[];
  brollSuggestions: BrollSuggestion[];
  editingPlan: EditingPlan;
}

// ── Helpers ──────────────────────────────────────

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function ScoreBadge({ score, max = 100 }: { score: number; max?: number }) {
  let color = 'bg-red-600';
  if (score >= 70) color = 'bg-green-600';
  else if (score >= 40) color = 'bg-yellow-600';

  return (
    <span className={`inline-block rounded-full px-3 py-1 text-sm font-bold text-white ${color}`}>
      {score}/{max}
    </span>
  );
}

function RiskBadge({ risk, t }: { risk: string; t: (key: string) => string }) {
  let color = 'bg-green-600';
  if (risk === 'high') color = 'bg-red-600';
  else if (risk === 'medium') color = 'bg-yellow-600';

  return (
    <span className={`inline-block rounded-full px-3 py-1 text-sm font-bold text-white ${color}`}>
      {t(`preview.risk.${risk}`)}
    </span>
  );
}

// ── Main Component ───────────────────────────────

function PreviewPage() {
  const { t } = useTranslation();
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();

  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [approving, setApproving] = useState(false);
  const [selectedHook, setSelectedHook] = useState(0);
  const [brollChecked, setBrollChecked] = useState<boolean[]>([]);
  const [revisionNotes, setRevisionNotes] = useState('');
  const [revising, setRevising] = useState(false);
  const [revisionError, setRevisionError] = useState<string | null>(null);

  const fetchAnalysis = useCallback(async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}/analysis`);
      if (!res.ok) {
        setError('notReady');
        return;
      }
      const data = (await res.json()) as AnalysisData;
      setAnalysis(data);
      setBrollChecked(data.brollSuggestions?.map(() => true) ?? []);
    } catch {
      setError('fetchError');
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  useEffect(() => {
    fetchAnalysis();
  }, [fetchAnalysis]);

  const handleApprove = async () => {
    if (!analysis) return;
    setApproving(true);

    try {
      const approvedPlan = {
        ...analysis,
        selectedHook: selectedHook as 0 | 1,
        brollSuggestions: analysis.brollSuggestions?.filter((_, i) => brollChecked[i]) ?? [],
      };

      const res = await fetch(`/api/jobs/${jobId}/analysis/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(approvedPlan),
      });

      if (res.ok) {
        navigate(`/result/${jobId}`);
      }
    } catch {
      // Silently fail
    } finally {
      setApproving(false);
    }
  };

  const handleRevise = async () => {
    if (!revisionNotes.trim()) return;
    setRevising(true);
    setRevisionError(null);

    try {
      const res = await fetch(`/api/jobs/${jobId}/analysis/revise`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes: revisionNotes }),
      });

      if (!res.ok) {
        const err = (await res.json()) as { error?: string };
        setRevisionError(err.error ?? 'Unknown error');
        return;
      }

      const data = (await res.json()) as AnalysisData;
      setAnalysis(data);
      setBrollChecked(data.brollSuggestions?.map(() => true) ?? []);
      setRevisionNotes('');
    } catch {
      setRevisionError('Network error');
    } finally {
      setRevising(false);
    }
  };

  const toggleBroll = (index: number) => {
    setBrollChecked((prev) => {
      const next = [...prev];
      next[index] = !next[index];
      return next;
    });
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center gap-4">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-gray-600 border-t-blue-400" />
        <p className="text-gray-400">{t('preview.loading')}</p>
      </div>
    );
  }

  if (error || !analysis) {
    return (
      <div className="flex flex-col items-center gap-4">
        <p className="text-red-400">{t('preview.errorLoading')}</p>
        <button
          onClick={() => navigate(`/analysis/${jobId}`)}
          className="rounded-xl border border-gray-700 bg-gray-800 px-6 py-2.5 text-gray-300 transition-colors hover:border-gray-500 hover:text-white"
        >
          {t('analysis.backToSettings')}
        </button>
      </div>
    );
  }

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-6 pb-12">
      <h2 className="text-center text-3xl font-bold text-white">{t('preview.title')}</h2>

      {/* Section 1 — Summary */}
      <Card title={t('preview.summary')}>
        <p className="mb-4 text-gray-300">{t('preview.videoAbout')}: {analysis.summary}</p>
        <div className="grid grid-cols-2 gap-3">
          <InfoRow label={t('preview.detectedGenre')} value={analysis.detectedGenre} />
          <InfoRow label={t('preview.targetAudience')} value={analysis.targetAudience} />
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">{t('preview.viralityScore')}:</span>
            <ScoreBadge score={analysis.viralityScore} />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">{t('preview.dropOffRisk')}:</span>
            <RiskBadge risk={analysis.retentionRisk} t={t} />
          </div>
        </div>
      </Card>

      {/* Section 2 — Hook Options */}
      <Card title={t('preview.hook')}>
        <div className="flex flex-col gap-3">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {analysis.hookOptions.map((option, i) => {
              const isSelected = selectedHook === i;
              return (
                <div
                  key={i}
                  className={`rounded-lg border p-3 transition-colors ${
                    isSelected
                      ? 'border-blue-500 bg-blue-900/20'
                      : 'border-gray-700 bg-gray-800/50'
                  }`}
                >
                  <div className="mb-1 flex items-center justify-between">
                    <span className="text-sm font-semibold text-gray-400">
                      {t('preview.hookOption')} {i + 1}
                    </span>
                    <ScoreBadge score={option.score} />
                  </div>
                  <p className="text-gray-200">"{option.text}"</p>
                  <p className="mt-1 text-xs text-gray-500">
                    {formatTime(option.start)} - {formatTime(option.end)}
                  </p>
                  <p className="mt-2 text-sm text-gray-400">
                    <span className="font-semibold text-gray-300">{t('preview.hookExplanation')}:</span>{' '}
                    {option.explanation}
                  </p>
                  {option.isWeaker && (
                    <p className="mt-1 text-xs text-yellow-400">{t('preview.hookWeaker')}</p>
                  )}
                  <button
                    onClick={() => setSelectedHook(i)}
                    className={`mt-3 w-full rounded-lg px-4 py-2 text-sm font-semibold transition-colors ${
                      isSelected
                        ? 'bg-blue-600 text-white'
                        : 'border border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-400 hover:text-white'
                    }`}
                  >
                    {isSelected ? t('preview.selectedHook') : t('preview.selectHook')}
                  </button>
                </div>
              );
            })}
          </div>

          {analysis.isWeakStart && (
            <div className="rounded-lg border border-yellow-700 bg-yellow-900/20 p-2 text-center text-sm text-yellow-300">
              {t('preview.weakStartWarning')}
            </div>
          )}
        </div>
      </Card>

      {/* Section 3 — Structure */}
      <Card title={t('preview.suggestedStructure')}>
        <div className="flex flex-col gap-2">
          {analysis.structure.sections.map((section, i) => (
            <div
              key={i}
              className="flex items-center justify-between rounded-lg border border-gray-700 bg-gray-800/50 px-4 py-2"
            >
              <div className="flex items-center gap-3">
                <SectionTypeBadge type={section.type} t={t} />
                <span className="text-gray-300">{section.description}</span>
              </div>
              <span className="text-sm text-gray-500">
                {formatTime(section.start)} - {formatTime(section.end)}
              </span>
            </div>
          ))}
        </div>
      </Card>

      {/* Section 4 — Strong Points */}
      {analysis.strongPoints.length > 0 && (
        <Card title={t('preview.strongPoints')}>
          <div className="flex flex-col gap-2">
            {analysis.strongPoints.map((point, i) => (
              <div key={i} className="rounded-lg border border-green-800/50 bg-green-900/10 p-3">
                <p className="text-gray-200">"{point.text}"</p>
                <div className="mt-1 flex items-center justify-between">
                  <span className="text-sm text-green-400">{point.reason}</span>
                  <span className="text-xs text-gray-500">
                    {formatTime(point.start)} - {formatTime(point.end)}
                  </span>
                </div>
                <span className="mt-1 inline-block rounded bg-green-800/30 px-2 py-0.5 text-xs text-green-300">
                  {t(`preview.suggestedAction.${point.suggestedAction}`, { defaultValue: point.suggestedAction })}
                </span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Section 5 — B-Roll Suggestions */}
      {analysis.brollSuggestions.length > 0 && (
        <Card title={t('preview.suggestedBroll')}>
          <div className="flex flex-col gap-2">
            {analysis.brollSuggestions.map((broll, i) => (
              <div
                key={i}
                className="flex items-start gap-3 rounded-lg border border-gray-700 bg-gray-800/50 p-3"
              >
                <input
                  type="checkbox"
                  checked={brollChecked[i] ?? true}
                  onChange={() => toggleBroll(i)}
                  className="mt-1 h-4 w-4 rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-blue-500"
                />
                <div className="flex-1">
                  <p className="text-gray-200">{broll.description}</p>
                  <p className="mt-1 text-sm text-gray-400">{broll.reason}</p>
                  <div className="mt-1 flex items-center gap-3">
                    <span className="text-xs text-gray-500">
                      {formatTime(broll.start)} - {formatTime(broll.end)}
                    </span>
                    <PriorityBadge priority={broll.priority} t={t} />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Section 7 — Editing Plan */}
      <Card title={t('preview.editingPlanSection')}>
        <div className="flex flex-col gap-3">
          <InfoRow
            label={t('preview.estimatedDuration')}
            value={`${analysis.editingPlan.estimatedDuration} ${t('settings.seconds')}`}
          />
          <InfoRow label={t('preview.cuts')} value={analysis.editingPlan.cuts} />
          <InfoRow label={t('preview.music')} value={analysis.editingPlan.music} />
          <InfoRow label={t('preview.subtitles')} value={analysis.editingPlan.subtitles} />

          {analysis.editingPlan.effects.length > 0 && (
            <div>
              <span className="text-sm text-gray-400">{t('preview.effects')}:</span>
              <div className="mt-1 flex flex-wrap gap-2">
                {analysis.editingPlan.effects.map((effect, i) => (
                  <span
                    key={i}
                    className="rounded bg-gray-700 px-2 py-1 text-xs text-gray-300"
                  >
                    {t(`preview.effectType.${effect.type}`, { defaultValue: effect.type })} @ {formatTime(effect.start)} — {effect.reason}
                  </span>
                ))}
              </div>
            </div>
          )}

          {analysis.editingPlan.cta && (
            <div className="rounded-lg border border-purple-800/50 bg-purple-900/10 p-3">
              <span className="text-sm font-semibold text-purple-400">{t('preview.cta')}</span>
              <p className="text-gray-200">"{analysis.editingPlan.cta.text}"</p>
              <p className="text-xs text-gray-500">
                {formatTime(analysis.editingPlan.cta.start)} - {formatTime(analysis.editingPlan.cta.end)} | {analysis.editingPlan.cta.type}
              </p>
            </div>
          )}
        </div>
      </Card>

      {/* Section 8 — Revision Notes */}
      <Card title={t('preview.haveFeedback')}>
        <textarea
          value={revisionNotes}
          onChange={(e) => setRevisionNotes(e.target.value)}
          placeholder={t('preview.revisionNotes')}
          disabled={revising}
          className="w-full rounded-lg border border-gray-700 bg-gray-800 p-3 text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none disabled:opacity-50"
          rows={3}
        />
        {revisionError && (
          <p className="mt-2 text-sm text-red-400">{revisionError}</p>
        )}
        <button
          onClick={handleRevise}
          disabled={revising || !revisionNotes.trim()}
          className="mt-3 rounded-xl bg-blue-600 px-6 py-2.5 font-semibold text-white transition-colors hover:bg-blue-500 disabled:opacity-50"
        >
          {revising ? (
            <span className="flex items-center gap-2">
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
              {t('preview.revising')}
            </span>
          ) : (
            t('preview.revise')
          )}
        </button>
      </Card>

      {/* Section 9 — Action Buttons */}
      <div className="flex items-center justify-center gap-4">
        <button
          onClick={() => navigate(`/settings/${jobId}`)}
          className="rounded-xl border border-gray-700 bg-gray-800 px-6 py-3 text-gray-300 transition-colors hover:border-gray-500 hover:text-white"
        >
          {t('preview.backToSettings')}
        </button>
        <button
          onClick={handleApprove}
          disabled={approving}
          className="rounded-xl bg-green-600 px-8 py-3 text-lg font-semibold text-white transition-colors hover:bg-green-500 disabled:opacity-50"
        >
          {approving ? t('preview.approving') : t('preview.approve')}
        </button>
      </div>
    </div>
  );
}

// ── Sub-components ───────────────────────────────

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 p-6">
      <h3 className="mb-4 text-xl font-semibold text-white">{title}</h3>
      {children}
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-sm text-gray-400">{label}:</span>
      <span className="text-gray-200">{value}</span>
    </div>
  );
}

function SectionTypeBadge({ type, t }: { type: string; t: (key: string, options?: Record<string, string>) => string }) {
  const colors: Record<string, string> = {
    hook: 'bg-yellow-700 text-yellow-200',
    setup: 'bg-blue-700 text-blue-200',
    content: 'bg-green-700 text-green-200',
    proof: 'bg-purple-700 text-purple-200',
    cta: 'bg-red-700 text-red-200',
  };
  const colorClass = colors[type] ?? 'bg-gray-700 text-gray-200';

  return (
    <span className={`rounded px-2 py-0.5 text-xs font-bold uppercase ${colorClass}`}>
      {t(`preview.sectionType.${type}`, { defaultValue: type })}
    </span>
  );
}

function PriorityBadge({ priority, t }: { priority: string; t: (key: string, options?: Record<string, string>) => string }) {
  const colors: Record<string, string> = {
    high: 'bg-red-800/30 text-red-300',
    medium: 'bg-yellow-800/30 text-yellow-300',
    low: 'bg-gray-700 text-gray-400',
  };
  const colorClass = colors[priority] ?? 'bg-gray-700 text-gray-400';

  return (
    <span className={`rounded px-2 py-0.5 text-xs ${colorClass}`}>
      {t(`preview.priority.${priority}`, { defaultValue: priority })}
    </span>
  );
}

export default PreviewPage;
