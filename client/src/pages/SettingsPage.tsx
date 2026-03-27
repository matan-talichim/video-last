import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useParams, useLocation, useNavigate } from 'react-router-dom';

interface VideoInfo {
  jobId: string;
  fileName: string;
  fileSize: number;
  duration: number;
  width: number;
  height: number;
  fps: number;
}

const VIDEO_TYPES = [
  'sponsored_ad',
  'brand_video',
  'product_marketing',
  'tiktok_reel',
  'story',
  'general',
] as const;

const DURATIONS = ['15', '30', '45', '60', '90', 'auto'] as const;

const BROLL_MODELS = [
  'kling_2_5',
  'veo_3_1_fast',
  'veo_3_1_quality',
  'sora_2',
  'wan_2_5',
  'seedance',
  'auto',
] as const;

const AI_BRAINS = ['claude_sonnet_4_6', 'gpt_5_4'] as const;

function formatFileSize(bytes: number): string {
  if (bytes >= 1073741824) return `${(bytes / 1073741824).toFixed(2)} GB`;
  if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(1)} MB`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function SettingsPage() {
  const { t } = useTranslation();
  const { jobId } = useParams<{ jobId: string }>();
  const location = useLocation();
  const navigate = useNavigate();

  const videoInfo = location.state as VideoInfo | null;

  const [videoType, setVideoType] = useState<(typeof VIDEO_TYPES)[number]>('general');
  const [targetDuration, setTargetDuration] = useState<(typeof DURATIONS)[number]>('auto');
  const [brollModel, setBrollModel] = useState<(typeof BROLL_MODELS)[number]>('auto');
  const [aiBrain, setAiBrain] = useState<(typeof AI_BRAINS)[number]>('claude_sonnet_4_6');
  const [template, setTemplate] = useState(t('settings.templates.general'));
  const [saving, setSaving] = useState(false);

  const handleVideoTypeChange = (type: (typeof VIDEO_TYPES)[number]) => {
    setVideoType(type);
    setTemplate(t(`settings.templates.${type}`));
  };

  const handleStartAnalysis = async () => {
    setSaving(true);

    try {
      const settings = {
        videoType,
        targetDuration,
        brollModel,
        aiBrain,
        template,
      };

      const res = await fetch(`/api/jobs/${jobId}/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings),
      });

      if (!res.ok) {
        throw new Error('Failed to save settings');
      }

      navigate(`/analysis/${jobId}`);
    } catch {
      setSaving(false);
    }
  };

  return (
    <div className="mx-auto max-w-3xl">
      <h2 className="mb-8 text-3xl font-bold text-white">{t('settings.title')}</h2>

      {/* Video Info Section */}
      {videoInfo && (
        <Section title={t('settings.videoInfo')}>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <InfoItem label={t('settings.fileName')} value={videoInfo.fileName} />
            <InfoItem label={t('settings.fileSize')} value={formatFileSize(videoInfo.fileSize)} />
            <InfoItem label={t('settings.duration')} value={formatDuration(videoInfo.duration)} />
            <InfoItem label={t('settings.resolution')} value={`${videoInfo.width}x${videoInfo.height}`} />
          </div>
        </Section>
      )}

      {/* Video Type */}
      <Section title={t('settings.videoType')}>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          {VIDEO_TYPES.map((type) => (
            <SelectButton
              key={type}
              label={t(`settings.videoTypes.${type}`)}
              selected={videoType === type}
              onClick={() => handleVideoTypeChange(type)}
            />
          ))}
        </div>
      </Section>

      {/* Target Duration */}
      <Section title={t('settings.targetDuration')}>
        <div className="flex flex-wrap gap-3">
          {DURATIONS.map((d) => (
            <SelectButton
              key={d}
              label={t(`settings.durations.${d}`)}
              selected={targetDuration === d}
              onClick={() => setTargetDuration(d)}
            />
          ))}
        </div>
      </Section>

      {/* B-Roll Model */}
      <Section title={t('settings.brollModel')}>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {BROLL_MODELS.map((model) => (
            <SelectButton
              key={model}
              label={t(`settings.brollModels.${model}`)}
              selected={brollModel === model}
              onClick={() => setBrollModel(model)}
            />
          ))}
        </div>
      </Section>

      {/* AI Brain */}
      <Section title={t('settings.aiBrain')}>
        <div className="flex flex-wrap gap-3">
          {AI_BRAINS.map((brain) => (
            <button
              key={brain}
              onClick={() => setAiBrain(brain)}
              className={`flex items-center gap-2.5 rounded-xl border px-5 py-3 font-medium transition-all ${
                aiBrain === brain
                  ? 'border-blue-500 bg-blue-600/20 text-blue-400'
                  : 'border-gray-700 bg-gray-800 text-gray-300 hover:border-gray-500'
              }`}
            >
              {brain === 'claude_sonnet_4_6' ? (
                <svg className="h-5 w-5" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="currentColor" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              ) : (
                <svg className="h-5 w-5" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2a10 10 0 110 20 10 10 0 010-20zm0 2a8 8 0 100 16 8 8 0 000-16zm-1 4h2v4h3l-4 4-4-4h3V8z" />
                </svg>
              )}
              {t(`settings.aiBrains.${brain}`)}
            </button>
          ))}
        </div>
      </Section>

      {/* Custom Template */}
      <Section title={t('settings.template')}>
        <textarea
          value={template}
          onChange={(e) => setTemplate(e.target.value)}
          placeholder={t('settings.templatePlaceholder')}
          rows={4}
          className="w-full resize-y rounded-xl border border-gray-700 bg-gray-800 px-4 py-3 text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
        />
      </Section>

      {/* Start Analysis Button */}
      <div className="mt-10 flex justify-center">
        <button
          onClick={handleStartAnalysis}
          disabled={saving}
          className="rounded-2xl bg-gradient-to-r from-blue-600 to-purple-600 px-12 py-4 text-xl font-bold text-white shadow-lg transition-all hover:from-blue-500 hover:to-purple-500 hover:shadow-xl disabled:opacity-50"
        >
          {saving ? t('settings.saving') : t('settings.startAnalysis')}
        </button>
      </div>
    </div>
  );
}

// ── Helper components ─────────────────────────────

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-8 rounded-xl border border-gray-800 bg-gray-900 p-6">
      <h3 className="mb-4 text-lg font-semibold text-gray-200">{title}</h3>
      {children}
    </div>
  );
}

function InfoItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-sm text-gray-500">{label}</p>
      <p className="mt-1 truncate font-medium text-white" title={value}>{value}</p>
    </div>
  );
}

function SelectButton({
  label,
  selected,
  onClick,
}: {
  label: string;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded-xl border px-4 py-2.5 text-sm font-medium transition-all ${
        selected
          ? 'border-blue-500 bg-blue-600/20 text-blue-400'
          : 'border-gray-700 bg-gray-800 text-gray-300 hover:border-gray-500'
      }`}
    >
      {label}
    </button>
  );
}

export default SettingsPage;
