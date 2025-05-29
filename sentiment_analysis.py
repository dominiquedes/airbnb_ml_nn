import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Read the grouped data
print("Reading grouped data...")
try:
    reviews_df = pd.read_csv('data/grouped_reviews.csv')
    listings_df = pd.read_csv('data/grouped_listings.csv')
except FileNotFoundError:
    print("Error: Could not find the grouped data files. Please run data_grouping.py first.")
    exit(1)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment scores
def get_sentiment_scores(text):
    if pd.isna(text):
        return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
    return analyzer.polarity_scores(str(text))

# Calculate sentiment scores for each review
print("Calculating sentiment scores...")
sentiment_scores = reviews_df['comments'].apply(get_sentiment_scores)
reviews_df['sentiment_compound'] = sentiment_scores.apply(lambda x: x['compound'])
reviews_df['sentiment_pos'] = sentiment_scores.apply(lambda x: x['pos'])
reviews_df['sentiment_neu'] = sentiment_scores.apply(lambda x: x['neu'])
reviews_df['sentiment_neg'] = sentiment_scores.apply(lambda x: x['neg'])

# Calculate average sentiment scores by performance group
print("\nCalculating average sentiment scores by group...")
group_sentiment = reviews_df.groupby('performance_group').agg({
    'sentiment_compound': 'mean',
    'sentiment_pos': 'mean',
    'sentiment_neu': 'mean',
    'sentiment_neg': 'mean'
}).round(3)

# Save mean sentiments to CSV
print("\nSaving mean sentiments to CSV...")
group_sentiment.to_csv('data/mean_sentiments.csv')
print("Mean sentiments saved to 'mean_sentiments.csv'")

print("\nAverage Sentiment Scores by Performance Group:")
print(group_sentiment)

# Create visualizations
plt.style.use('default')  # Using default style instead of seaborn
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Compound Sentiment Score by Group
sns.boxplot(data=reviews_df, x='performance_group', y='sentiment_compound', ax=ax1)
ax1.set_title('Distribution of Sentiment Scores by Performance Group')
ax1.set_xlabel('Performance Group')
ax1.set_ylabel('Compound Sentiment Score')
ax1.set_ylim(-1, 1)

# Plot 2: Average Sentiment Components by Group
sentiment_components = group_sentiment[['sentiment_pos', 'sentiment_neu', 'sentiment_neg']]
sentiment_components.plot(kind='bar', ax=ax2)
ax2.set_title('Average Sentiment Components by Performance Group')
ax2.set_xlabel('Performance Group')
ax2.set_ylabel('Average Score')
ax2.legend(['Positive', 'Neutral', 'Negative'])
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('sentiment_analysis_results.png')
print("\nVisualization saved as 'sentiment_analysis_results.png'")

# Additional analysis: Word count and sentiment correlation
reviews_df['word_count'] = reviews_df['comments'].str.split().str.len()
print("\nCorrelation between word count and sentiment:")
print(reviews_df[['word_count', 'sentiment_compound']].corr())

# Save the sentiment analysis results
reviews_df.to_csv('data/reviews_with_sentiment.csv', index=False)
print("\nSentiment analysis complete! Results saved to 'reviews_with_sentiment.csv'") 