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

# Generate a recommendation using Gemini with JSON output
def generate_recommendation(user_preferences, similar_movies, df):
    context = f"User preferences: {user_preferences}\n\nSimilar movies based on content:"
    movie_list = []
    for i, movie in similar_movies.iterrows():
        release_year = movie['Release Year'] if pd.notna(movie['Release Year']) else 'N/A'
        movie_details = {
            "title": movie['Title'],
            "release_year": release_year,
            "director": movie['Director'],
            "genre": movie['Genre'],
            "cast": movie['Cast'],
            "plot": movie['Plot']
        }
        movie_list.append(movie_details)
        context += f"\n\n{i+1}. {movie['Title']} ({release_year})"
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
  "similarity_score": A number between 0-100 representing how well this matches the user's preferences
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
        # Optionally include a few similar movies in the response
        recommendation_json["similar_movies"] = [movie_list[i] for i in range(min(3, len(movie_list)))]
        return recommendation_json
    except json.JSONDecodeError as e:
        fallback_json = {
            "error": "Failed to parse JSON response",
            "raw_response": response.text,
            "similar_movies": movie_list[:3] if len(movie_list) >= 3 else movie_list
        }
        return fallback_json

# Function to search for movies by genre keywords
def search_by_genre(df, genre_keywords, top_n=5):
    matched_movies = []
    # Convert keywords to lowercase for case-insensitive matching
    genre_keywords = [keyword.lower() for keyword in genre_keywords]
    for idx, row in df.iterrows():
        score = 0
        if pd.notna(row['Genre']):
            genre_text = row['Genre'].lower()
            for keyword in genre_keywords:
                if keyword in genre_text:
                    score += 1
        if score > 0:
            matched_movies.append((idx, score))
    # Sort by matching score (highest first)
    matched_movies.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, score in matched_movies[:top_n]]
    
    if not top_indices:
        # Fallback: return the most recent movies if no match is found
        return df.sort_values('Release Year', ascending=False).head(top_n)
    
    return df.iloc[top_indices][['Title', 'Release Year', 'Director', 'Genre', 'Plot', 'Cast']].reset_index(drop=True)

# Main recommendation function using only user preferences (no reference movie)
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
    
    # Extract genre keywords from the user's preferences
    result["process_log"].append("Extracting genre keywords from user preferences")
    genre_keywords = re.findall(
        r'\b(?:action|comedy|drama|sci-fi|thriller|horror|romance|fantasy|animation|documentary|biography|musical|crime|war|western|history|sport|mystery)\b', 
        user_preferences.lower()
    )
    result["genre_keywords_found"] = genre_keywords
    
    if genre_keywords:
        similar_movies = search_by_genre(df, genre_keywords, top_n)
        result["process_log"].append(f"Found {len(similar_movies)} movies matching genre keywords: {', '.join(genre_keywords)}")
    else:
        result["process_log"].append("No specific genres identified. Using most recent movies")
        similar_movies = df.sort_values('Release Year', ascending=False).head(top_n)
        similar_movies = similar_movies[['Title', 'Release Year', 'Director', 'Genre', 'Plot', 'Cast']].reset_index(drop=True)
    
    # Generate recommendation
    result["process_log"].append("Generating personalized recommendation")
    try:
        recommendation = generate_recommendation(user_preferences, similar_movies, df)
        result["status"] = "success"
        result["recommendation"] = recommendation
        result["process_log"].append("Recommendation generated successfully")
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Failed to generate recommendation: {str(e)}"
        return result
    
    return result

