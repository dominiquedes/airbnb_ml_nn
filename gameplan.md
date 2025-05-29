# 🏡 Airbnb Listings & Reviews Analysis Project

## 🎯 Goal  
Compare the performance of a **Random Forest** model versus a **Neural Network** to predict Airbnb listing prices based on selected features.

## 📁 Required Files  
- `listings.csv`  
- `reviews.csv`

## 🗂️ 1. Data Grouping

### Listings (`data/listings.csv`)  
- Define a performance metric (e.g., price, number of reviews, occupancy, overall rating).  
- Use quantiles (top 33%, middle 33%, bottom 33%) to categorize listings into:  
  - Top-performing  
  - Mid-performing  
  - Bottom-performing

### Reviews (`data/reviews.csv`)  
- Group reviews by `listing_id`  
- Join with `listings.csv` to assign each review to the corresponding performance group

## 📊 2. Exploratory Data Analysis (EDA)

### 🧼 Data Cleaning & Preprocessing  
- Handle missing values and inconsistencies  
- Encode categorical features:
  - One-hot encode `room_type`, `location`
  - Add binary flags for features like `wifi`, `pet_friendly`
- Normalize or scale numerical features

### 🧠 Feature Engineering  
Use the following features (and other relevant attributes as needed):

*Full feature list available in `listings.csv`:*  
`listing_id,listing_url,scrape_id,last_scraped,source,name,description,neighborhood_overview,picture_url,host_id,host_url,host_name,host_since,host_location,host_about,host_response_time,host_response_rate,host_acceptance_rate,host_is_superhost,host_thumbnail_url,host_picture_url,host_neighbourhood,host_listings_count,host_total_listings_count,host_verifications,host_has_profile_pic,host_identity_verified,neighbourhood,neighbourhood_cleansed,neighbourhood_group_cleansed,latitude,longitude,property_type,room_type,accommodates,bathrooms,bathrooms_text,bedrooms,beds,amenities,price,minimum_nights,maximum_nights,minimum_minimum_nights,maximum_minimum_nights,minimum_maximum_nights,maximum_maximum_nights,minimum_nights_avg_ntm,maximum_nights_avg_ntm,calendar_updated,has_availability,availability_30,availability_60,availability_90,availability_365,calendar_last_scraped,number_of_reviews,number_of_reviews_ltm,number_of_reviews_l30d,availability_eoy,number_of_reviews_ly,estimated_occupancy_l365d,estimated_revenue_l365d,first_review,last_review,review_scores_rating,review_scores_accuracy,review_scores_cleanliness,review_scores_checkin,review_scores_communication,review_scores_location,review_scores_value,license,instant_bookable,calculated_host_listings_count,calculated_host_listings_count_entire_homes,calculated_host_listings_count_private_rooms,calculated_host_listings_count_shared_rooms,reviews_per_month,amenities_count,is_superhost_numeric,performance_score,performance_group`

*Full features in `reviews.csv`:*
`performance_group,sentiment_compound,sentiment_pos,sentiment_neu,sentiment_neg`

accommodates
bedrooms
bathrooms
room_type
property_type
amenities_count
latitude
longitude
neighbourhood_cleansed
review_scores_rating
review_scores_cleanliness
review_scores_value
host_is_superhost_numeric
number_of_reviews
reviews_per_month
availability_365
minimum_nights
performance_score

why: They provide a balance of physical property characteristcs, location information, quality metrics, host information, demand indicaters, performance metrics  

## 🤖 3. Model Analysis by Performance Group

### Models to Use  
- **Random Forest**  
- **Neural Network**

### For Each Group  
- Train both models using the selected features  
- Predict the target variable (e.g., performance class or price)  
- Evaluate model performance using:
  - Accuracy  
  - Precision / Recall  
  - F1 Score

## 💬 4. Sentiment Analysis on Reviews

- For each listing in every performance group (Top, Mid, Bottom), select two reviews  
- Apply sentiment analysis tools (e.g., VADER, TextBlob, BERT)  
- Analyze polarity and sentiment scores  
- Compare sentiment trends across performance tiers

## 📈 5. Comparative Analysis

- Visual comparison of model results using:
  - Confusion Matrix  
  - ROC Curve  
  - Bar charts or heatmaps for performance metrics  
- Visualize sentiment score distributions across performance groups


## Frontend
- Convert listings and reviews  CSV into a JSON 
#### JSON formation
- reviews --> listing_id -> review_id, date, reviewer name, comments
- listings --> listing_id -> name, host_id, host_name, neighbourhood_group, neighbourhood, latitude, longitude, room_type, price, minimum_nights, number_of_reviews, last_review, reviews_per_month, calculated_host_listings_count, availability_365, number_of_reviews_ltm, license
#### How to display the information
- html and css to display listsings
* When a listing is clicked, display the listing information along with its reviews.
