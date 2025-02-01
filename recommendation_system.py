import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dot, Flatten, Dense

# Create watch history DataFrame
watch_history = pd.DataFrame({
    "user_id": [1, 1, 2, 2, 3, 3, 4, 4],
    "video_id": [101, 102, 101, 103, 104, 102, 101, 105],
    "watch_time": [120, 150, 200, 300, 100, 180, 150, 210],
    "timestamp": [
        "2025-01-01 10:00:00", "2025-01-01 10:10:00", "2025-01-02 14:00:00",
        "2025-01-02 14:30:00", "2025-01-03 11:00:00", "2025-01-03 11:20:00",
        "2025-01-04 09:00:00", "2025-01-04 09:30:00"
    ]
})

# Create video metadata DataFrame
video_metadata = pd.DataFrame({
    "video_id": [101, 102, 103, 104, 105],
    "title": [
        "How to Learn Python", "The Future of AI", "Top 10 Tech Gadgets",
        "Healthy Living Tips", "Ultimate Guide to Data Science"
    ],
    "description": [
        "Python tutorial for beginners", "Exploring AI advancements and its impact",
        "Review of top 10 gadgets of the year", "Advice on how to live a healthy life",
        "Learn data science from scratch"
    ],
    "tags": [
        "python, tutorial, programming", "AI, machine learning, technology",
        "gadgets, tech, reviews", "health, fitness, lifestyle", "data science, machine learning, analytics"
    ],
    "category": ["Education", "Technology", "Technology", "Health", "Education"],
    "likes": [500, 300, 700, 400, 600],
    "dislikes": [20, 30, 50, 10, 40]
})

# Create user preferences DataFrame
user_preferences = pd.DataFrame({
    "user_id": [1, 2, 3, 4],
    "liked_videos": ["101,102", "103,101", "104,102", "105,101"],
    "disliked_videos": ["105", "104", "101", "103"]
})

# Merge watch history and user preferences
df = pd.merge(watch_history, user_preferences, on='user_id', how='left').fillna("")

# Content-based filtering using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(video_metadata['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_similar_videos(video_id, top_n=5):
    idx = video_metadata[video_metadata['video_id'] == video_id].index[0]
    similar_videos = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
    top_videos = [video_metadata.iloc[i[0]]['video_id'] for i in similar_videos[1:top_n+1]]
    return top_videos

# Collaborative filtering using SVD
user_video_matrix = df.pivot(index='user_id', columns='video_id', values='watch_time').fillna(0)
user_video_sparse = csr_matrix(user_video_matrix)

U, sigma, Vt = svds(user_video_sparse, k=2)
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt)

def recommend_videos(user_id, top_n=5):
    idx = user_video_matrix.index.get_loc(user_id)
    sorted_indices = np.argsort(predicted_ratings[idx])[::-1]
    return [user_video_matrix.columns[i] for i in sorted_indices[:top_n]]

# Deep learning recommendation using Neural Collaborative Filtering
# Normalize IDs (Ensure 0-based indexing)
watch_history["user_id"] -= watch_history["user_id"].min()
watch_history["video_id"] -= watch_history["video_id"].min()

# Number of unique users & videos
num_users = watch_history["user_id"].nunique()
num_videos = watch_history["video_id"].nunique()

# Define Model Inputs
user_input = Input(shape=(1,), dtype=tf.int32, name="user_input")
video_input = Input(shape=(1,), dtype=tf.int32, name="video_input")

# Embeddings (Ensure Correct Input Size)
user_embedding = Embedding(input_dim=num_users + 1, output_dim=50, name="user_embedding")(user_input)
video_embedding = Embedding(input_dim=num_videos + 1, output_dim=50, name="video_embedding")(video_input)

# Compute Dot Product
dot_product = Dot(axes=1, name="dot_product")([Flatten()(user_embedding), Flatten()(video_embedding)])
output = Dense(1, activation="sigmoid", name="output")(dot_product)

# Build & Compile Model
model = Model(inputs=[user_input, video_input], outputs=output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Prepare Training Data
user_ids = watch_history["user_id"].astype(np.int32).values
video_ids = watch_history["video_id"].astype(np.int32).values
ratings = (watch_history["watch_time"] > 100).astype(np.int32).values  # Binary rating: Watched or Not

# Train Model
model.fit([user_ids, video_ids], ratings, epochs=5, batch_size=4, verbose=1)

# Function to Predict Recommendations
def deep_recommend_videos(user_id, top_n=5):
    normalized_user_id = user_id - watch_history["user_id"].min()
    
    user_input_data = np.full((num_videos,), normalized_user_id, dtype=np.int32)
    video_input_data = np.arange(num_videos, dtype=np.int32)
    
    predictions = model.predict([user_input_data, video_input_data]).flatten()
    top_video_indices = predictions.argsort()[-top_n:][::-1]
    
    return video_metadata.iloc[top_video_indices]["video_id"].values

# Test Recommendation
print(deep_recommend_videos(user_id=1, top_n=3))

# Hybrid recommendation system
def hybrid_recommend(user_id, video_id, top_n=5):
    content_based = get_similar_videos(video_id, top_n // 2)
    collaborative = recommend_videos(user_id, top_n // 2)
    deep_learning = deep_recommend_videos(user_id, top_n // 2)
    
    final_recommendations = list(set(content_based + collaborative + list(deep_learning)))
    return final_recommendations[:top_n]

# Example usage
print("Hybrid Recommendations:", hybrid_recommend(user_id=1, video_id=101, top_n=5))
