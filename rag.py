from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import re
import json

# Configure the Gemini API
genai.configure(api_key="AIzaSyCUHMZpf2pJ9ezf-8w2sqJbUbvxpDH9ts8")

# Load and preprocess the movie dataset
def load_movie_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    df['Plot'] = df['Plot'].fillna('')
    df['Genre'] = df['Genre'].fillna('')
    df['Director'] = df['Director'].fillna('')
    df['Cast'] = df['Cast'].fillna('')
    df['combined_features'] = df.apply(
        lambda row: f"{row['Title']} {row['Plot']} {row['Genre']} {row['Director']} {row['Cast']} {row['Origin/Ethnicity']}", 
        axis=1
    )
    return df

# Create embeddings for the movies using TF-IDF
def create_movie_embeddings(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    return tfidf_matrix, tfidf

# Extract genre keywords from user preferences
def extract_genre_keywords(user_preferences):
    return re.findall(
        r'\b(?:action|comedy|drama|sci-fi|thriller|horror|romance|fantasy|animation|documentary|biography|musical|crime|war|western|history|sport|mystery)\b', 
        user_preferences.lower()
    )

# Find movies similar to user preferences using both genre matching and TF-IDF embeddings
def find_similar_movies_hybrid(user_preferences, df, tfidf_matrix, tfidf, top_n=5):
    # Extract genre keywords from user preferences
    genre_keywords = extract_genre_keywords(user_preferences)
    
    # Transform user preferences into the same vector space
    user_vector = tfidf.transform([user_preferences])
    
    # Calculate cosine similarity between user preferences and all movies
    cosine_sim = cosine_similarity(user_vector, tfidf_matrix).flatten()
    
    if genre_keywords:
        # Calculate genre match score for all movies
        genre_scores = np.zeros(len(df))
        
        for idx, row in df.iterrows():
            if pd.notna(row['Genre']):
                genre_text = row['Genre'].lower()
                for keyword in genre_keywords:
                    if keyword in genre_text:
                        genre_scores[idx] += 1
        
        # Normalize genre scores to range 0-1
        if genre_scores.max() > 0:
            genre_scores = genre_scores / genre_scores.max()
        
        # Create a combined score with higher weight for genre matches
        # Genre weight: 0.7, Similarity weight: 0.3
        combined_scores = (0.7 * genre_scores) + (0.3 * cosine_sim)
    else:
        # If no genre keywords, use only cosine similarity
        combined_scores = cosine_sim
    
    # Get indices of top movies by combined score
    top_indices = combined_scores.argsort()[-top_n:][::-1]
    
    # Return the similar movies and their combined scores
    return df.iloc[top_indices][['Title', 'Release Year', 'Director', 'Genre', 'Plot', 'Cast']].reset_index(drop=True), combined_scores[top_indices]

# Generate a recommendation using Gemini with JSON output
def generate_recommendation(user_preferences, similar_movies, similarity_scores, df, genre_keywords=None):
    context = f"User preferences: {user_preferences}\n"
    if genre_keywords:
        context += f"Detected genre interests: {', '.join(genre_keywords)}\n"
    context += "\nSimilar movies based on content:"
    
    movie_list = []
    for i, (_, movie) in enumerate(similar_movies.iterrows()):
        release_year = movie['Release Year'] if pd.notna(movie['Release Year']) else 'N/A'
        similarity_pct = round(similarity_scores[i] * 100, 2)
        
        movie_details = {
            "title": movie['Title'],
            "release_year": release_year,
            "director": movie['Director'],
            "genre": movie['Genre'],
            "cast": movie['Cast'],
            "plot": movie['Plot'],
            "similarity_score": similarity_pct
        }
        movie_list.append(movie_details)
        context += f"\n\n{i+1}. {movie['Title']} ({release_year}) - Match Score: {similarity_pct}%"
        context += f"\nDirector: {movie['Director']}"
        context += f"\nGenre: {movie['Genre']}"
        context += f"\nCast: {movie['Cast']}"
        context += f"\nPlot: {movie['Plot']}"
    
    prompt = f"""
Based on the user's preferences ("{user_preferences}") and the following similar movies, recommend the best movie from the list with a personalized explanation of why they would enjoy it. Focus on matching movie attributes with user preferences.

{context}

Provide your recommendation as a valid JSON object with the following structure:
{{
  "recommended_movie": {{
    "title": "Movie Title",
    "release_year": "Year",
    "director": "Director Name",
    "cast": "Main Cast Members",
    "genre": "Movie Genre"
  }},
  "recommendation_reason": "Detailed personalized explanation of why the user would enjoy this movie based on their preferences",
  "match_score": A number between 0-100 representing how well this matches the user's preferences
}}

Ensure the output is a valid, parseable JSON object with no additional text before or after.
    """
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    
    try:
        response_text = response.text
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "", 1)
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        recommendation_json = json.loads(response_text)
        # Include top similar movies in the response
        recommendation_json["similar_movies"] = movie_list
        if genre_keywords:
            recommendation_json["detected_genres"] = genre_keywords
        return recommendation_json
    except json.JSONDecodeError as e:
        fallback_json = {
            "error": "Failed to parse JSON response",
            "raw_response": response.text,
            "similar_movies": movie_list
        }
        return fallback_json

# Main recommendation function using hybrid approach
def recommend_movies_rag(csv_path, user_preferences, top_n=5):
    result = {
        "status": "processing",
        "user_preferences": user_preferences,
        "process_log": []
    }
    
    # Load movie data
    result["process_log"].append("Loading movie data")
    try:
        df = load_movie_data(csv_path)
        result["process_log"].append(f"Loaded {len(df)} movies successfully")
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Failed to load movie data: {str(e)}"
        return result
    
    # Create embeddings
    result["process_log"].append("Creating movie embeddings")
    try:
        tfidf_matrix, tfidf = create_movie_embeddings(df)
        result["process_log"].append("Embeddings created successfully")
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Failed to create embeddings: {str(e)}"
        return result
    
    # Extract genre keywords from user preferences
    genre_keywords = extract_genre_keywords(user_preferences)
    result["genre_keywords_found"] = genre_keywords
    if genre_keywords:
        result["process_log"].append(f"Detected genre keywords: {', '.join(genre_keywords)}")
    else:
        result["process_log"].append("No specific genres detected in user preferences")
    
    # Use hybrid approach (genre priority + TF-IDF embeddings) to find similar movies
    result["process_log"].append("Finding movies using hybrid approach (genre priority + content similarity)")
    try:
        similar_movies, similarity_scores = find_similar_movies_hybrid(user_preferences, df, tfidf_matrix, tfidf, top_n)
        result["process_log"].append(f"Found {len(similar_movies)} relevant movies using hybrid approach")
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Failed to find similar movies: {str(e)}"
        return result
    
    # Generate recommendation
    result["process_log"].append("Generating personalized recommendation")
    try:
        recommendation = generate_recommendation(user_preferences, similar_movies, similarity_scores, df, genre_keywords)
        result["status"] = "success"
        result["recommendation"] = recommendation
        result["process_log"].append("Recommendation generated successfully")
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Failed to generate recommendation: {str(e)}"
        return result
    
    return result