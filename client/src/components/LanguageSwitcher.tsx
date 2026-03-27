import { useTranslation } from 'react-i18next';

function LanguageSwitcher() {
  const { i18n, t } = useTranslation();
  const currentLang = i18n.language;

  function toggleLanguage() {
    const nextLang = currentLang === 'he' ? 'en' : 'he';
    i18n.changeLanguage(nextLang);
  }

  const label = currentLang === 'he' ? t('language.en') : t('language.he');

  return (
    <button
      onClick={toggleLanguage}
      className="rounded-lg border border-gray-700 bg-gray-800 px-4 py-2 text-sm text-gray-300 transition-colors hover:bg-gray-700 hover:text-white"
    >
      {label}
    </button>
  );
}

export default LanguageSwitcher;
