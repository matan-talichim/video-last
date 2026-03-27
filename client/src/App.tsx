import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout.js';
import HomePage from './pages/HomePage.js';
import SettingsPage from './pages/SettingsPage.js';
import AnalysisPage from './pages/AnalysisPage.js';

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/settings/:jobId" element={<SettingsPage />} />
        <Route path="/analysis/:jobId" element={<AnalysisPage />} />
      </Routes>
    </Layout>
  );
}

export default App;
