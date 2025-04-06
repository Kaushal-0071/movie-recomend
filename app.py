import pandas as pd
import numpy as np
from ast import literal_eval



import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import json



import google.generativeai as genai
app = Flask(__name__)
CORS(app)
#collabrative filtering recommender
class hybrid_recomsys:

    def __init__(self, metadata_path, links_small_path, ratings_small_path):

        self.metadata = pd.read_csv(metadata_path)
        self.ratings_small = pd.read_csv(ratings_small_path)
        self.links_small = pd.read_csv(links_small_path)
        self.id_map = pd.read_csv(links_small_path)[['movieId', 'tmdbId']]

    def get_small_metadata(self):

        """Function to clean the data and get small dataset from metadata

        We need to use small dataset to avoide the expesive computational power"""

        self.metadata['vote_count'] = self.metadata[self.metadata['vote_count'].notnull()]['vote_count'].astype('int')
        self.metadata['vote_average'] = self.metadata[self.metadata['vote_average'].notnull()]['vote_average'].astype('int')
        self.metadata = self.metadata.drop([19730, 29503, 35587]) #some incomplete data
        self.metadata['id'] = self.metadata['id'].astype('int')
        self.metadata['year'] = pd.to_datetime(self.metadata['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

        self.links_small = self.links_small[self.links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

        self.small_data = self.metadata[self.metadata['id'].isin(self.links_small)]
        #print(self.small_data.shape)

        self.small_data['tagline'] = self.small_data['tagline'].fillna('')
        self.small_data['description'] = self.small_data['overview'] + self.small_data['tagline']
        self.small_data['description'] = self.small_data['description'].fillna('')

    def get_cosine_sim(self):

        """Function to callculate cosine similarity based on the  movie description"""

        tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=1, stop_words='english')
        tfidf_matrix = tf.fit_transform(self.small_data['description'])
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        #print(tfidf_matrix.shape)

    def get_SVD_model(self):

        """"Function to train the SVD model with small data

        The model predict rating from userId input, the accuracy indicated by RMSE

        """

        reader = Reader()
        data = Dataset.load_from_df(self.ratings_small[['userId', 'movieId', 'rating']], reader)

        # Split the dataset into training and testing sets
        trainset, testset = train_test_split(data, test_size=0.2)

        # Initialize the SVD model
        self.svd = SVD()

        # Fit the model to the training data
        self.svd.fit(trainset)

        # Perform cross-validation and evaluate the model
        results = cross_validate(self.svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

        # Access the results (RMSE and MAE for each fold)
        for fold_num in range(5):
            print(f"Fold {fold_num + 1}: RMSE = {results['test_rmse'][fold_num]}, MAE = {results['test_mae'][fold_num]}")

        # Optionally, you can calculate the mean RMSE and MAE across all folds
        mean_rmse = results['test_rmse'].mean()
        mean_mae = results['test_mae'].mean()
        print(f"Mean RMSE across folds: {mean_rmse}")
        print(f"Mean MAE across folds: {mean_mae}")

    def convert_int(self, x):

        """Function to convert x to int """
        try:
            return int(x)
        except:
            return np.nan

    def get_idmap(self):
        """Function to convert x to int """
        self.small_data = self.small_data.reset_index()
        self.indices = pd.Series(self.small_data.index, index=self.small_data['title'])

        self.id_map['tmdbId'] = self.id_map['tmdbId'].apply(self.convert_int)
        self.id_map.columns = ['movieId', 'id']
        self.id_map = self.id_map.merge(self.small_data[['title', 'id']], on='id').set_index('title')
        self.indices_map = self.id_map.set_index('id')

    def prep_hybrid(self):

        """Function to prepared the SVD model and cosin similarity matrix to make hybrid recommedation"""

        self.get_small_metadata()
        self.get_cosine_sim()
        self.get_SVD_model()
        self.get_idmap()

    def add_new_rating(self, userId, movieId, rating):
        """Add a new user rating and update the dataset."""
        new_rating = pd.DataFrame([[userId, movieId, rating]], columns=['userId', 'movieId', 'rating'])
        self.ratings_small = pd.concat([self.ratings_small, new_rating], ignore_index=True)

    def retrain_svd(self):
        """Retrain the SVD model after adding new ratings."""
        reader = Reader()
        data = Dataset.load_from_df(self.ratings_small[['userId', 'movieId', 'rating']], reader)

        trainset, testset = train_test_split(data, test_size=0.2)
        self.svd = SVD()
        self.svd.fit(trainset)

    def update_and_recommend(self, userId, title, rating, display=10):
        """Update ratings, retrain model, and get recommendations."""
        movie_id = self.id_map.loc[title]['movieId']

        # Add new rating
        self.add_new_rating(userId, movie_id, rating)

        # Retrain model with new data
        self.retrain_svd()

        # Get recommendations
        return self.main(userId, title, display)


    def main(self, userId, title, display=10):


        """Function to make hybrid recommedation

        Args:
            userId(int): the user ID
            title(str): the movie that user interested

        Return:
            movies(object): the data table, which the recommeded movies

        """

        idx = self.indices[title]
        tmdbId = self.id_map.loc[title]['id']
        #print(idx)
        movie_id = self.id_map.loc[title]['movieId']

        sim_scores = list(enumerate(self.cosine_sim[int(idx)]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:26]
        movie_indices = [i[0] for i in sim_scores]

        movies = self.small_data.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
        movies['est'] = movies['id'].apply(lambda x: self.svd.predict(userId, self.indices_map.loc[x]['movieId']).est)
        movies = movies.sort_values('est', ascending=False)

        return movies.head(display)


#genre recommender
class KnowledgeRecommender:
    
    def __init__(self, database):
        path_database = database
        self.df = pd.read_csv(path_database, low_memory=False)
    
    def convert_int(self, x):
        try:
            return int(x)
        except:
            return 0
        
    def get_release_year(self, df):
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['year'] = df['release_date'].apply(lambda x: str(x).split('-')[0] if pd.notnull(x) else np.nan)
        df['year'] = df['year'].apply(self.convert_int)
        return df
    
    def get_genre(self, df):
        df['genres'] = df['genres'].fillna('[]')
        df['genres'] = df['genres'].apply(literal_eval)
        df['genres'] = df['genres'].apply(lambda x: [i['name'].lower() for i in x] if isinstance(x, list) else [])
        s = df.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'genre'
        df = df.join(s)
        return df
    
    def precondition(self, df, quantile_num=0.80, runtime=[10, 200]):
        self.m = df['vote_count'].quantile(quantile_num)
        self.C = df['vote_average'].mean()
        df_q_movies = df[(df['runtime'] >= runtime[0]) & (df['runtime'] <= runtime[1])]
        df_q_movies = df_q_movies[df_q_movies['vote_count'] >= self.m]
        return df_q_movies
    
    def weighted_rating(self, df):
        v = df['vote_count']
        R = df['vote_average']
        weight_score = (v / (v + self.m) * R) + (self.m / (self.m + v) * self.C)
        return weight_score
    
    def main(self, genre, see_top=25, low_time=10, high_time=200, low_year=1800, high_year=2024):
        self.df_q_movies = self.precondition(self.df, runtime=[low_time, high_time])
        self.df_q_movies = self.get_release_year(self.df_q_movies)
        self.df_q_movies = self.get_genre(self.df_q_movies)
        self.df_q_movies['score'] = self.df_q_movies.apply(self.weighted_rating, axis=1)
        self.df_q_movies = self.df_q_movies.sort_values('score', ascending=False)
        
        movies = self.df_q_movies.copy()
        movies = movies[(movies['genre'] == genre.lower()) & 
                         (movies['runtime'] >= low_time) & 
                         (movies['runtime'] <= high_time) & 
                         (movies['year'] >= low_year) & 
                         (movies['year'] <= high_year)]
        
        movies = movies[['title', 'genre', 'year', 'runtime', 'vote_average', 'vote_count', 'score']]
        return movies.sort_values('score', ascending=False).head(see_top)









metadata_path = r"movies_metadata.csv"
ratings_small_path = r"ratings_small.csv"
links_small_path = r"links_small.csv"
credits = r'credits.csv'
keywords = r'keywords.csv'


A1 = hybrid_recomsys(metadata_path, links_small_path, ratings_small_path)
A1.prep_hybrid()

preference_scored= KnowledgeRecommender(metadata_path)

#content based recommender
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0] #its function that find index value from table
    distances = similarity[movie_index] #measuring distance of array

    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6] 
    
    recommended_movie = []
    recommended_movie_posters = []
    for i in movies_list:
    #now we are try to fetch the poster of movie with movie id
        movie_id = movies.iloc[i[0]].movie_id

        recommended_movie.append(movies.iloc[i[0]].title)
        #fetch poster from API
        #recommended_movie_posters.append(fetch_poste(movie_id))
    return recommended_movie

# assign .pkl file to the frontend to show the list of titles of movies
movies_dict = pickle.load(open('movie_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)
    
#import statemnt for similarity
similarity = pickle.load(open('similarity.pkl','rb'))
print("done")



@app.route('/content-based', methods=['POST'])
def content_based():
    data = request.get_json()
    if not data or 'movies' not in data:
        return jsonify({'error': 'Please provide a list of movies in the JSON body with key "movies".'}), 400
    
    movies_list = data['movies']
    if not isinstance(movies_list, list):
        return jsonify({'error': 'Movies should be provided as a list.'}), 400
    
    try:
        all_recommendations = []
        for movie in movies_list:
            rec = recommend(movie)
            all_recommendations=rec+all_recommendations
        return jsonify({'recommended_movies': all_recommendations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500







@app.route('/recommend_genre', methods=['POST'])
def recommend_movies():
    try:
        data = request.get_json()
        genres = data.get('genres', [])
        see_top = data.get('see_top', 20)
       
        
        if not genres or not isinstance(genres, list):
            return jsonify({"error": "Invalid or missing genres. Provide a list of genres."}), 400

        # Get movies for all genres and concatenate
        all_movies = pd.DataFrame()
        for genre in genres:
            top_movies = preference_scored.main(genre, see_top=see_top)
            all_movies = pd.concat([all_movies, top_movies])
        
        # Remove duplicates, shuffle, and return top 20 random movies
        all_movies = all_movies.drop_duplicates().sample(n=min(20, len(all_movies)), random_state=42)
        
        return jsonify(all_movies.to_dict(orient='records'))

    except Exception as e:
        return jsonify({"error": str(e)}), 500






    
@app.route('/user_recommend', methods=['POST'])
def user_recommend():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        movies_list = data.get('movies')

        if not user_id or not movies_list:
            return jsonify({"error": "Missing user_id or movies list."}), 400

        if not isinstance(movies_list, list):
            return jsonify({"error": "Movies should be provided as a list."}), 400

        all_recommendations = pd.DataFrame()  # Initialize an empty DataFrame

        # Process each movie, skipping if an error occurs
        for title in movies_list:
            try:
                rec = A1.main(userId=user_id, title=title)
                all_recommendations = pd.concat([all_recommendations, rec])
            except Exception as movie_error:
                print(f"Error fetching recommendation for '{title}': {movie_error}")  # Log the error
                continue  # Skip this movie and move to the next one

        # Remove duplicates (if any)
        all_recommendations = all_recommendations.drop_duplicates()

        return jsonify(all_recommendations.to_dict(orient='records'))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

from rag import recommend_movies_rag

@app.route('/rag-recommend', methods=['POST'])
def rag_recommend():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Please provide a prompt in the JSON body with key 'prompt'."}), 400
    
    user_preferences = data['prompt']
    csv_path = "wiki_movie_plots_deduped.csv"  # Ensure this CSV is in the correct path
    recommendation_result = recommend_movies_rag(csv_path, user_preferences)
    
    if recommendation_result.get("status") != "success":
        return jsonify({"error": recommendation_result.get("error", "Unknown error"), 
                        "process_log": recommendation_result.get("process_log", [])}), 500
    
    return jsonify(recommendation_result)


if __name__ == '__main__':
    app.run()