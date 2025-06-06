document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultsContainer = document.getElementById('results');
    let map;
    let marker;
    let autocomplete;
    let geocoder;

    // Initialize Google Maps and Places
    function initMap() {
        // Create map centered on New York City
        map = new google.maps.Map(document.getElementById('map'), {
            center: { lat: 40.7128, lng: -74.0060 },
            zoom: 12,
            styles: [
                {
                    "featureType": "poi",
                    "stylers": [{ "visibility": "off" }]
                }
            ]
        });

        // Initialize marker
        marker = new google.maps.Marker({
            map: map,
            draggable: true,
            animation: google.maps.Animation.DROP
        });

        // Initialize geocoder
        geocoder = new google.maps.Geocoder();

        // Initialize autocomplete with enhanced options
        autocomplete = new google.maps.places.Autocomplete(
            document.getElementById('address'),
            { 
                types: ['address'],
                componentRestrictions: { country: 'us' },
                bounds: new google.maps.LatLngBounds(
                    { lat: 40.4774, lng: -74.2591 }, // SW bounds of NYC
                    { lat: 40.9176, lng: -73.7004 }  // NE bounds of NYC
                ),
                fields: ['address_components', 'geometry', 'formatted_address']
            }
        );

        // Handle place selection with detailed address components
        autocomplete.addListener('place_changed', () => {
            const place = autocomplete.getPlace();
            if (!place.geometry) {
                alert('Please select a valid address from the suggestions.');
                return;
            }

            // Update map and marker
            map.setCenter(place.geometry.location);
            map.setZoom(16);
            marker.setPosition(place.geometry.location);

            // Update hidden coordinate fields
            document.getElementById('latitude').value = place.geometry.location.lat();
            document.getElementById('longitude').value = place.geometry.location.lng();

            // Extract and set neighborhood if available
            const addressComponents = place.address_components;
            for (const component of addressComponents) {
                if (component.types.includes('sublocality_level_1')) {
                    const neighborhood = component.long_name;
                    const neighborhoodSelect = document.getElementById('neighbourhood');
                    const option = Array.from(neighborhoodSelect.options).find(opt => 
                        opt.text.toLowerCase().includes(neighborhood.toLowerCase())
                    );
                    if (option) {
                        neighborhoodSelect.value = option.value;
                    }
                }
            }

            // Add a small bounce animation to the marker
            marker.setAnimation(google.maps.Animation.BOUNCE);
            setTimeout(() => {
                marker.setAnimation(null);
            }, 750);
        });

        // Handle marker drag with reverse geocoding
        marker.addListener('dragend', () => {
            const position = marker.getPosition();
            document.getElementById('latitude').value = position.lat();
            document.getElementById('longitude').value = position.lng();
            
            // Reverse geocode to update address
            geocoder.geocode({ location: position }, (results, status) => {
                if (status === 'OK' && results[0]) {
                    document.getElementById('address').value = results[0].formatted_address;
                    
                    // Update neighborhood if available
                    const addressComponents = results[0].address_components;
                    for (const component of addressComponents) {
                        if (component.types.includes('sublocality_level_1')) {
                            const neighborhood = component.long_name;
                            const neighborhoodSelect = document.getElementById('neighbourhood');
                            const option = Array.from(neighborhoodSelect.options).find(opt => 
                                opt.text.toLowerCase().includes(neighborhood.toLowerCase())
                            );
                            if (option) {
                                neighborhoodSelect.value = option.value;
                            }
                        }
                    }
                }
            });
        });

        // Add click listener to map
        map.addListener('click', (event) => {
            marker.setPosition(event.latLng);
            document.getElementById('latitude').value = event.latLng.lat();
            document.getElementById('longitude').value = event.latLng.lng();
            
            // Reverse geocode the clicked location
            geocoder.geocode({ location: event.latLng }, (results, status) => {
                if (status === 'OK' && results[0]) {
                    document.getElementById('address').value = results[0].formatted_address;
                }
            });
        });
    }

    // Initialize map when the page loads
    initMap();

    // Add input event listener for address field
    const addressInput = document.getElementById('address');
    addressInput.addEventListener('input', () => {
        // Clear coordinates when user starts typing
        document.getElementById('latitude').value = '';
        document.getElementById('longitude').value = '';
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Validate coordinates
        const latitude = document.getElementById('latitude').value;
        const longitude = document.getElementById('longitude').value;
        
        if (!latitude || !longitude) {
            alert('Please select a valid address from the suggestions.');
            return;
        }
        
        // Show loading state
        const submitButton = form.querySelector('button[type="submit"]');
        const originalText = submitButton.textContent;
        submitButton.textContent = 'Predicting...';
        submitButton.disabled = true;

        try {
            // Get form data with only required features
            const formData = {
                room_type: form.room_type.value,
                neighbourhood_cleansed: document.getElementById('closest_neighbourhood').value,
                property_type: form.property_type.value,
                host_is_superhost: form.host_is_superhost.checked ? 1 : 0,
                accommodates: parseInt(form.accommodates.value),
                bathrooms: parseFloat(form.bathrooms.value),
                bedrooms: parseInt(form.bedrooms.value),
                beds: parseInt(form.beds.value),
                latitude: parseFloat(latitude),
                longitude: parseFloat(longitude),
                minimum_nights: parseInt(form.minimum_nights.value),
                number_of_reviews: 25.94,
                review_scores_rating: 4.72,
                review_scores_cleanliness: 4.66,
                review_scores_value: 4.64,
                // Count selected amenities
                essential_amenities: document.querySelectorAll('input[name="essential_amenities"]:checked').length,
                safety_amenities: document.querySelectorAll('input[name="safety_amenities"]:checked').length,
                luxury_amenities: document.querySelectorAll('input[name="luxury_amenities"]:checked').length,
                outdoor_amenities: document.querySelectorAll('input[name="outdoor_amenities"]:checked').length
            };

            // Log the data types of all numerical fields for debugging
            console.log('Data types check:', {
                accommodates: typeof formData.accommodates,
                bathrooms: typeof formData.bathrooms,
                bedrooms: typeof formData.bedrooms,
                beds: typeof formData.beds,
                latitude: typeof formData.latitude,
                longitude: typeof formData.longitude,
                minimum_nights: typeof formData.minimum_nights,
                number_of_reviews: typeof formData.number_of_reviews,
                review_scores_rating: typeof formData.review_scores_rating
            });

            // Make API request
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to get prediction');
            }

            const data = await response.json();
            
            if (data.status === 'success') {
                // Update results
                document.getElementById('rf-price').textContent = `$${data.prediction.toFixed(2)}`;
                
                // Show results
                resultsContainer.style.display = 'block';
                
                // Scroll to results
                resultsContainer.scrollIntoView({ behavior: 'smooth' });
            } else {
                throw new Error(data.message || 'Failed to get prediction');
            }

        } catch (error) {
            alert(error.message);
        } finally {
            // Reset button state
            submitButton.textContent = originalText;
            submitButton.disabled = false;
        }
    });
}); 