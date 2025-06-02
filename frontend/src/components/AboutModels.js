import React from 'react';

const AboutModels = () => {
  return (
    <div>
      <h1 className="title">About Our Models</h1>
      
      <div>
        <section className="card">
          <h2 className="subtitle">Neural Network Model</h2>
          <p className="text">
            Our primary model is a sophisticated neural network that has been trained on thousands of
            Airbnb listings. The model considers multiple factors to make accurate price predictions:
          </p>
          <ul className="text" style={{ listStyle: 'disc', paddingLeft: '1.5rem' }}>
            <li>Property characteristics (size, rooms, amenities)</li>
            <li>Location data (neighborhood, coordinates)</li>
            <li>Host information (superhost status, response rate)</li>
            <li>Review metrics (scores, number of reviews)</li>
            <li>Availability and booking patterns</li>
          </ul>
        </section>

        <section className="card">
          <h2 className="subtitle">Model Performance</h2>
          <p className="text">
            Our model has been rigorously tested and validated:
          </p>
          <ul className="text" style={{ listStyle: 'disc', paddingLeft: '1.5rem' }}>
            <li>Test RMSE: ~$77</li>
            <li>Test MAE: ~$49</li>
            <li>Cross-validated performance metrics</li>
            <li>Regular updates with new data</li>
          </ul>
        </section>

        <section className="card">
          <h2 className="subtitle">Key Features</h2>
          <p className="text">
            The model takes into account these important features:
          </p>
          <div className="grid">
            <div>
              <h3 className="subtitle">Property Features</h3>
              <ul className="text" style={{ listStyle: 'disc', paddingLeft: '1.5rem' }}>
                <li>Number of bedrooms</li>
                <li>Number of bathrooms</li>
                <li>Room type</li>
                <li>Property type</li>
                <li>Amenities count</li>
              </ul>
            </div>
            <div>
              <h3 className="subtitle">Location & Host Features</h3>
              <ul className="text" style={{ listStyle: 'disc', paddingLeft: '1.5rem' }}>
                <li>Neighborhood</li>
                <li>Latitude/Longitude</li>
                <li>Superhost status</li>
                <li>Review scores</li>
                <li>Availability</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="card">
          <h2 className="subtitle">Model Architecture</h2>
          <p className="text">
            Our neural network architecture includes:
          </p>
          <ul className="text" style={{ listStyle: 'disc', paddingLeft: '1.5rem' }}>
            <li>Multiple dense layers with dropout for regularization</li>
            <li>Batch normalization for stable training</li>
            <li>LeakyReLU activation functions</li>
            <li>Early stopping to prevent overfitting</li>
            <li>Feature engineering and selection</li>
          </ul>
        </section>
      </div>
    </div>
  );
};

export default AboutModels; 