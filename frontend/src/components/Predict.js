import React, { useState } from 'react';

const Predict = () => {
  const [formData, setFormData] = useState({
    accommodates: '',
    bedrooms: '',
    bathrooms: '',
    room_type: 'Entire home/apt',
    property_type: 'Apartment',
    amenities_count: '',
    latitude: '',
    longitude: '',
    neighbourhood: '',
    minimum_nights: '',
    availability_365: ''
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    // TODO: Implement prediction logic
    console.log('Form submitted:', formData);
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  return (
    <div>
      <h1 className="title">Price Prediction</h1>
      <div className="card">
        <form onSubmit={handleSubmit}>
          <div className="grid">
            <div className="form-group">
              <label className="form-label">Accommodates</label>
              <input
                type="number"
                name="accommodates"
                value={formData.accommodates}
                onChange={handleChange}
                className="form-input"
                required
              />
            </div>
            <div className="form-group">
              <label className="form-label">Bedrooms</label>
              <input
                type="number"
                name="bedrooms"
                value={formData.bedrooms}
                onChange={handleChange}
                className="form-input"
                required
              />
            </div>
            <div className="form-group">
              <label className="form-label">Bathrooms</label>
              <input
                type="number"
                name="bathrooms"
                value={formData.bathrooms}
                onChange={handleChange}
                className="form-input"
                required
              />
            </div>
            <div className="form-group">
              <label className="form-label">Room Type</label>
              <select
                name="room_type"
                value={formData.room_type}
                onChange={handleChange}
                className="form-input"
              >
                <option value="Entire home/apt">Entire home/apt</option>
                <option value="Private room">Private room</option>
                <option value="Shared room">Shared room</option>
                <option value="Hotel room">Hotel room</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Property Type</label>
              <select
                name="property_type"
                value={formData.property_type}
                onChange={handleChange}
                className="form-input"
              >
                <option value="Apartment">Apartment</option>
                <option value="House">House</option>
                <option value="Villa">Villa</option>
                <option value="Condo">Condo</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Number of Amenities</label>
              <input
                type="number"
                name="amenities_count"
                value={formData.amenities_count}
                onChange={handleChange}
                className="form-input"
                required
              />
            </div>
          </div>
          
          <div style={{ textAlign: 'right', marginTop: '1rem' }}>
            <button type="submit" className="button button-primary">
              Predict Price
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Predict; 