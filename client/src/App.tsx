import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout.js';
import HomePage from './pages/HomePage.js';

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
      </Routes>
    </Layout>
  );
}

export default App;
