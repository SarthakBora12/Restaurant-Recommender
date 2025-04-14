# app.py
import os
import gdown

# Download filtered_reviews_data.csv
if not os.path.exists("filtered_reviews_data.csv"):
    gdown.download("https://drive.google.com/file/d/1C0Lyfm_2QfiUvfnLzQFMapC9bLLSF6D1/view?usp=drive_link", "filtered_reviews_data.csv", quiet=False)

# Download filtered_business_data.csv
if not os.path.exists("filtered_business_data.csv"):
    gdown.download("https://drive.google.com/file/d/1BHFazN4LSm-9NJen4WJbgX2Q5X7rzoUT/view?usp=drive_link", "filtered_business_data.csv", quiet=False)

# Download svd_model.pkl
if not os.path.exists("svd_model.pkl"):
    gdown.download("https://drive.google.com/file/d/1FkFBBovxKLWUVlAR9_2-BkJqsO_m57BE/view?usp=drive_link", "svd_model.pkl", quiet=False)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from surprise import SVD
import pickle
import warnings

warnings.filterwarnings('ignore')

# === Load model ===
with open("svd_model.pkl", "rb") as f:
    model = pickle.load(f)

# === Load datasets ===
business_df = pd.read_csv("filtered_business_data.csv")
reviews_df = pd.read_csv("filtered_reviews_data.csv")

# === Recommendation function ===
def hybrid_recommend_for_user(user_id, business_df, reviews_df, model, top_n=5, city=None, category_filter=None):
    user_reviewed = reviews_df[reviews_df['user_id'] == user_id]['business_id'].unique()
    candidates = business_df[~business_df['business_id'].isin(user_reviewed)]

    if city:
        candidates = candidates[candidates['city'].str.lower() == city.lower()]
    if category_filter:
        candidates = candidates[candidates['categories'].str.contains(category_filter, case=False, na=False)]

    sentiment_avg = reviews_df.groupby('business_id')['vader_sentiment'].mean().to_dict()
    recommendations = []

    for _, row in candidates.iterrows():
        business_id = row['business_id']
        name = row['name']
        try:
            pred = model.predict(user_id, business_id)
            predicted_rating = pred.est
            sentiment_score = sentiment_avg.get(business_id, 0)
            final_score = 0.6 * predicted_rating + 0.4 * sentiment_score
            recommendations.append({
                'Restaurant': name,
                'Predicted Rating': predicted_rating,
                'Sentiment Score': sentiment_score,
                'Final Score': final_score
            })
        except:
            continue

    rec_df = pd.DataFrame(recommendations).sort_values(by='Final Score', ascending=False).head(top_n)
    return rec_df.reset_index(drop=True)

# === Streamlit UI ===
st.set_page_config(page_title="Restaurant Recommender", layout="wide")
st.title("🍽️ Restaurant Recommendation System")
st.markdown("Get personalized restaurant recommendations based on your preferences, location, and review sentiment.")

# Sidebar filters
st.sidebar.header("🔎 Filters")
user_id = st.sidebar.selectbox("Select User ID", reviews_df['user_id'].unique())
city = st.sidebar.selectbox("Select City", sorted(business_df['city'].unique()))
category = st.sidebar.text_input("Filter by Category (e.g., Sushi, Coffee)", "")
top_n = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

# Button to get recommendations
if st.sidebar.button("Get Recommendations"):
    rec_df = hybrid_recommend_for_user(
        user_id=user_id,
        business_df=business_df,
        reviews_df=reviews_df,
        model=model,
        top_n=top_n,
        city=city,
        category_filter=category if category.strip() else None
    )

    st.subheader("📋 Top Recommendations")
    st.dataframe(rec_df)

    # Visualization
    st.subheader("📊 Score Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = rec_df['Restaurant']
    ax.bar(x, rec_df['Predicted Rating'], label='Predicted Rating')
    ax.bar(x, rec_df['Sentiment Score'], label='Sentiment Score', alpha=0.7)
    ax.bar(x, rec_df['Final Score'], label='Final Score', alpha=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Score')
    plt.title("Score Breakdown for Recommended Restaurants")
    plt.legend()
    st.pyplot(fig)

    st.success("✅ Recommendations generated based on collaborative filtering and review sentiment!")

else:
    st.info("Select a user, city, and filters on the left and click 'Get Recommendations'.")
