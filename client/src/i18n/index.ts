import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import he from './he.json';
import en from './en.json';

const savedLanguage = typeof window !== 'undefined'
  ? localStorage.getItem('ai-editor-lang') ?? 'he'
  : 'he';

i18n.use(initReactI18next).init({
  resources: {
    he: { translation: he },
    en: { translation: en },
  },
  lng: savedLanguage,
  fallbackLng: 'en',
  interpolation: {
    escapeValue: false,
  },
});

i18n.on('languageChanged', (lng: string) => {
  localStorage.setItem('ai-editor-lang', lng);
  document.documentElement.lang = lng;
  document.documentElement.dir = lng === 'he' ? 'rtl' : 'ltr';
});

// Set initial direction
if (typeof document !== 'undefined') {
  document.documentElement.lang = savedLanguage;
  document.documentElement.dir = savedLanguage === 'he' ? 'rtl' : 'ltr';
}

export default i18n;
