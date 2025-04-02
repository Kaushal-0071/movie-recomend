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
    # Read the CSV file with the specific columns
    # "Release Year,Title,Origin/Ethnicity,Director,Cast,Genre,Wiki Page,Plot"
    df = pd.read_csv(csv_path)
    
    # Ensure column names are correctly formatted (in case there are spaces)
    df.columns = [col.strip() for col in df.columns]
    
    # Fill NaN values
    df['Plot'] = df['Plot'].fillna('')
    df['Genre'] = df['Genre'].fillna('')
    df['Director'] = df['Director'].fillna('')
    df['Cast'] = df['Cast'].fillna('')
    
    # Create a text field that combines important movie attributes for embedding
    df['combined_features'] = df.apply(
        lambda row: f"{row['Title']} {row['Plot']} {row['Genre']} {row['Director']} {row['Cast']} {row['Origin/Ethnicity']}", 
        axis=1
    )
    
    return df

# Create embeddings for the movies
def create_movie_embeddings(df):
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Create TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    return tfidf_matrix, tfidf

# Find similar movies based on content
def find_similar_movies(movie_title, df, tfidf_matrix, top_n=5):
    # Find the index of the movie with the given title
    if movie_title not in df['Title'].values:
        similar_titles = df['Title'][df['Title'].str.contains(movie_title, case=False)]
        if len(similar_titles) > 0:
            movie_title = similar_titles.iloc[0]
        else:
            return None, "Movie not found in the database."
    
    movie_idx = df[df['Title'] == movie_title].index[0]
    
    # Compute cosine similarity between the movie and all other movies
    cosine_sim = cosine_similarity(tfidf_matrix[movie_idx], tfidf_matrix).flatten()
    
    # Get indices of similar movies (excluding the movie itself)
    similar_indices = cosine_sim.argsort()[::-1][1:top_n+1]
    
    # Get the details of similar movies
    similar_movies = df.iloc[similar_indices][['Title', 'Release Year', 'Director', 'Genre', 'Plot', 'Cast']].reset_index(drop=True)
    
    return similar_movies, None

# Generate a recommendation using Gemini with JSON output
def generate_recommendation(user_preferences, similar_movies, df):
    # Create context for Gemini
    context = f"User preferences: {user_preferences}\n\nSimilar movies based on content:"
    
    # Create a list of movie details for the context
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
        
        # Also add text version for the prompt
        context += f"\n\n{i+1}. {movie['Title']} ({release_year})"
        context += f"\nDirector: {movie['Director']}"
        context += f"\nGenre: {movie['Genre']}"
        context += f"\nCast: {movie['Cast']}"
        context += f"\nPlot: {movie['Plot']}"
    
    # Define the prompt for JSON output
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
    
    # Generate recommendation using Gemini
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    
    # Try to parse the response as JSON
    try:
        # Strip any markdown code block indicators if present
        response_text = response.text
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "", 1)
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        recommendation_json = json.loads(response_text)
        
        # Add the similar movies to the JSON response
        recommendation_json["similar_movies"] = [movie_list[i] for i in range(min(3, len(movie_list)))]
        
        return recommendation_json
    except json.JSONDecodeError as e:
        # Return a fallback JSON if parsing fails
        fallback_json = {
            "error": "Failed to parse JSON response",
            "raw_response": response.text,
            "similar_movies": movie_list[:3] if len(movie_list) >= 3 else movie_list
        }
        return fallback_json

# Function to search for movies by genre keywords
def search_by_genre(df, genre_keywords, top_n=5):
    matched_movies = []
    
    # Convert all genre keywords to lowercase for case-insensitive matching
    genre_keywords = [keyword.lower() for keyword in genre_keywords]
    
    # Create a score for each movie based on matching genres
    for idx, row in df.iterrows():
        score = 0
        if pd.notna(row['Genre']):
            genre_text = row['Genre'].lower()
            for keyword in genre_keywords:
                if keyword in genre_text:
                    score += 1
        
        if score > 0:
            matched_movies.append((idx, score))
    
    # Sort by score (descending)
    matched_movies.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N matches
    top_indices = [idx for idx, score in matched_movies[:top_n]]
    
    if not top_indices:
        # If no genre matches, return the most popular/highest rated movies
        # Since we don't have ratings, use the most recent movies as a fallback
        return df.sort_values('Release Year', ascending=False).head(top_n)
    
    return df.iloc[top_indices][['Title', 'Release Year', 'Director', 'Genre', 'Plot', 'Cast']].reset_index(drop=True)

# Main function to run the movie recommendation system with JSON output
def recommend_movies(csv_path, user_preferences, reference_movie=None, top_n=5):
    # Create a result dictionary to track the process
    result = {
        "status": "processing",
        "user_preferences": user_preferences,
        "reference_movie": reference_movie,
        "process_log": []
    }
    
    # Load movie data
    print("Loading movie data...")
    result["process_log"].append("Loading movie data")
    try:
        df = load_movie_data(csv_path)
        result["process_log"].append(f"Loaded {len(df)} movies successfully")
        print(f"Loaded {len(df)} movies.")
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Failed to load movie data: {str(e)}"
        return result
    
    # Create embeddings
    print("Creating movie embeddings...")
    result["process_log"].append("Creating movie embeddings")
    try:
        tfidf_matrix, tfidf = create_movie_embeddings(df)
        result["process_log"].append("Embeddings created successfully")
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Failed to create embeddings: {str(e)}"
        return result
    
    # Find similar movies
    similar_movies = None
    if reference_movie:
        print(f"Finding movies similar to '{reference_movie}'...")
        result["process_log"].append(f"Finding movies similar to '{reference_movie}'")
        similar_movies, error = find_similar_movies(reference_movie, df, tfidf_matrix, top_n)
        if error:
            result["status"] = "error"
            result["error"] = error
            return result
        result["process_log"].append(f"Found {len(similar_movies)} similar movies")
    else:
        # If no reference movie, extract genre keywords from user preferences
        print("No reference movie provided. Searching by genre keywords...")
        result["process_log"].append("No reference movie provided. Searching by genre keywords")
        genre_keywords = re.findall(r'\b(?:action|comedy|drama|sci-fi|thriller|horror|romance|fantasy|animation|documentary|biography|musical|crime|war|western|history|sport|mystery)\b', 
                        user_preferences.lower())
        
        result["genre_keywords_found"] = genre_keywords
        
        if genre_keywords:
            similar_movies = search_by_genre(df, genre_keywords, top_n)
            result["process_log"].append(f"Found {len(similar_movies)} movies matching genre keywords: {', '.join(genre_keywords)}")
        else:
            # If no genres found, just get the most recent movies
            print("No specific genres identified. Using most recent movies...")
            result["process_log"].append("No specific genres identified. Using most recent movies")
            similar_movies = df.sort_values('Release Year', ascending=False).head(top_n)
            similar_movies = similar_movies[['Title', 'Release Year', 'Director', 'Genre', 'Plot', 'Cast']].reset_index(drop=True)
            result["process_log"].append(f"Selected {len(similar_movies)} most recent movies")
    
    # Generate recommendation
    print("Generating personalized recommendation...")
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

# Example usage
if __name__ == "__main__":
    csv_path = "wiki_movie_plots_deduped.csv"  # Path to your movie CSV file
    
    # Example: Recommend a horror movie without gore
    user_preferences = "a horror movie without gore"
    
    recommendation_result = recommend_movies(
        csv_path=csv_path,
        user_preferences=user_preferences
    )
    
    # Print the result as formatted JSON
    print("\n" + "="*50)
    print("MOVIE RECOMMENDATION (JSON OUTPUT)")
    print("="*50)
    print(json.dumps(recommendation_result, indent=2))
    print("="*50)