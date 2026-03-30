import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout.js';
import HomePage from './pages/HomePage.js';
import SettingsPage from './pages/SettingsPage.js';
import AnalysisPage from './pages/AnalysisPage.js';
import PreviewPage from './pages/PreviewPage.js';
import ResultPage from './pages/ResultPage.js';

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/settings/:jobId" element={<SettingsPage />} />
        <Route path="/analysis/:jobId" element={<AnalysisPage />} />
        <Route path="/preview/:jobId" element={<PreviewPage />} />
        <Route path="/result/:jobId" element={<ResultPage />} />
      </Routes>
    </Layout>
  );
}

export default App;
