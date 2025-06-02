import React from 'react';

const Home = () => {
  return (
    <div>
      <h1 className="title">Airbnb Price Prediction</h1>
      <div className="card">
        <h2 className="subtitle">Welcome to Our Price Prediction Tool</h2>
        <p className="text">
          This application helps you predict Airbnb listing prices using advanced machine learning models.
          Our models take into account various factors such as location, amenities, and property features
          to provide accurate price predictions.
        </p>
        <div className="grid">
          <div className="card">
            <h3 className="subtitle">Try Our Predictor</h3>
            <p className="text">
              Get instant price predictions for your Airbnb listing by providing some basic information
              about your property.
            </p>
            <a href="/predict" className="button button-primary">
              Go to Predictor
            </a>
          </div>
          <div className="card">
            <h3 className="subtitle">Learn About Our Models</h3>
            <p className="text">
              Discover how our machine learning models work and what factors they consider when
              making predictions.
            </p>
            <a href="/about-models" className="button button-secondary">
              Learn More
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home; 