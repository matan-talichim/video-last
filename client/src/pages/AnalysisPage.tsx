import { useTranslation } from 'react-i18next';
import { useParams, useNavigate } from 'react-router-dom';

function AnalysisPage() {
  const { t } = useTranslation();
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();

  return (
    <div className="flex flex-col items-center gap-8">
      <h2 className="text-3xl font-bold text-white">{t('analysis.title')}</h2>

      <div className="rounded-xl border border-gray-800 bg-gray-900 px-10 py-8 text-center">
        {/* Animated spinner */}
        <div className="mb-6 flex justify-center">
          <div className="h-12 w-12 animate-spin rounded-full border-4 border-gray-700 border-t-blue-500" />
        </div>
        <p className="text-lg text-gray-400">{t('analysis.placeholder')}</p>
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

export default AnalysisPage;
