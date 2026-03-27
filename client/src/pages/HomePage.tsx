import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

function HomePage() {
  const { t } = useTranslation();
  const [serverStatus, setServerStatus] = useState<'loading' | 'ok' | 'error'>('loading');

  useEffect(() => {
    fetch('/api/health')
      .then((res) => res.json())
      .then((data: { status?: string }) => {
        setServerStatus(data.status === 'ok' ? 'ok' : 'error');
      })
      .catch(() => {
        setServerStatus('error');
      });
  }, []);

  return (
    <div className="flex flex-col items-center gap-10">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-white">{t('home.welcome')}</h2>
        <p className="mt-3 text-lg text-gray-400">{t('home.description')}</p>
      </div>

      <div className="flex flex-col items-center gap-4 sm:flex-row">
        <button className="rounded-xl bg-blue-600 px-8 py-4 text-lg font-semibold text-white transition-colors hover:bg-blue-500">
          {t('home.uploadButton')}
        </button>
        <span className="text-gray-500">{t('home.or')}</span>
        <button className="rounded-xl bg-purple-600 px-8 py-4 text-lg font-semibold text-white transition-colors hover:bg-purple-500">
          {t('home.createButton')}
        </button>
      </div>

      <div className="rounded-xl border border-gray-800 bg-gray-900 px-6 py-4">
        <div className="flex items-center gap-3">
          <span className="text-gray-400">{t('home.status')}:</span>
          {serverStatus === 'loading' && (
            <span className="text-gray-500">...</span>
          )}
          {serverStatus === 'ok' && (
            <span className="flex items-center gap-2 text-green-400">
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-green-400" />
              {t('home.statusOk')}
            </span>
          )}
          {serverStatus === 'error' && (
            <span className="flex items-center gap-2 text-red-400">
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-red-400" />
              {t('home.statusError')}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

export default HomePage;
