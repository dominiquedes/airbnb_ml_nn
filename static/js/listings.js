// Global variables
let allListings = [];
let currentPage = 1;
const listingsPerPage = 12;
let filteredListings = [];

// Fetch listings when page loads
document.addEventListener('DOMContentLoaded', () => {
    fetchListings();
});

// Fetch listings from the server
async function fetchListings() {
    try {
        const response = await fetch('/api/listings');
        if (!response.ok) {
            throw new Error('Failed to fetch listings');
        }
        allListings = await response.json();
        filteredListings = [...allListings];
        displayListings();
    } catch (error) {
        console.error('Error fetching listings:', error);
        document.getElementById('listings-grid').innerHTML = '<p class="error">Failed to load listings. Please try again later.</p>';
    }
}

// Display listings in the grid
function displayListings() {
    const grid = document.getElementById('listings-grid');
    const startIndex = (currentPage - 1) * listingsPerPage;
    const endIndex = startIndex + listingsPerPage;
    const pageListings = filteredListings.slice(startIndex, endIndex);

    grid.innerHTML = pageListings.map(listing => `
        <div class="listing-card">
            <div class="listing-image">
                <img src="${listing.picture_url || '/static/images/no-image.jpg'}" alt="${listing.name}">
            </div>
            <div class="listing-info">
                <h3>${listing.name}</h3>
                <p class="listing-price">$${listing.price}/night</p>
                <p class="listing-type">${listing.room_type} in ${listing.neighbourhood_cleansed}</p>
                <div class="listing-details">
                    <span>${listing.bedrooms} beds</span>
                    <span>${listing.bathrooms} baths</span>
                    <span>${listing.accommodates} guests</span>
                </div>
                <div class="listing-rating">
                    <span class="stars">★</span>
                    <span>${listing.review_scores_rating || 'No reviews'}</span>
                </div>
            </div>
        </div>
    `).join('');

    updatePagination();
}

// Filter listings based on selected criteria
function filterListings() {
    const neighbourhood = document.getElementById('neighbourhood').value;
    const roomType = document.getElementById('room_type').value;
    const priceRange = document.getElementById('price_range').value;
    const minRating = parseFloat(document.getElementById('min_rating').value);

    filteredListings = allListings.filter(listing => {
        // Neighbourhood filter
        if (neighbourhood && listing.neighbourhood_cleansed !== neighbourhood) {
            return false;
        }

        // Room type filter
        if (roomType && listing.room_type !== roomType) {
            return false;
        }

        // Price range filter
        if (priceRange) {
            const [min, max] = priceRange.split('-').map(Number);
            if (max) {
                if (listing.price < min || listing.price > max) {
                    return false;
                }
            } else {
                // Handle "400+" case
                if (listing.price < min) {
                    return false;
                }
            }
        }

        // Rating filter
        if (minRating > 0 && (!listing.review_scores_rating || listing.review_scores_rating < minRating)) {
            return false;
        }

        return true;
    });

    currentPage = 1;
    displayListings();
}

// Update pagination controls
function updatePagination() {
    const totalPages = Math.ceil(filteredListings.length / listingsPerPage);
    document.getElementById('page-info').textContent = `Page ${currentPage} of ${totalPages}`;
    document.getElementById('prev-page').disabled = currentPage === 1;
    document.getElementById('next-page').disabled = currentPage === totalPages;
}

// Change page
function changePage(delta) {
    const totalPages = Math.ceil(filteredListings.length / listingsPerPage);
    currentPage = Math.max(1, Math.min(totalPages, currentPage + delta));
    displayListings();
} 