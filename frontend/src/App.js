import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Home from './components/Home';
import Predict from './components/Predict';
import AboutModels from './components/AboutModels';

function App() {
  return (
    <Router>
      <div>
        {/* Navigation */}
        <nav className="nav">
          <div className="container nav-content">
            <div>
              <Link to="/" className="nav-link">
                Airbnb Price Predictor
              </Link>
            </div>
            <div className="nav-links">
              <Link to="/" className="nav-link">
                Home
              </Link>
              <Link to="/predict" className="nav-link">
                Predict
              </Link>
              <Link to="/about-models" className="nav-link">
                About Models
              </Link>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="main-content">
          <div className="container">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/predict" element={<Predict />} />
              <Route path="/about-models" element={<AboutModels />} />
            </Routes>
          </div>
        </main>

        {/* Footer */}
        <footer className="footer">
          <div className="container">
            <p>© 2024 Airbnb Price Predictor. All rights reserved.</p>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;
