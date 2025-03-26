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
app = Flask(__name__)

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




class MergeCleanData:
    
    def __init__(self, metadata, credits, keywords):

        self.df = pd.read_csv(metadata)
        self.cred_df = pd.read_csv(credits)
        self.key_df = pd.read_csv(keywords)

        
    def clean_as_int(self, x):
        
        """Function to convert 'x' to integers, if can not, return Nan"""
        
        try:
            return int(x)
        except:
            return np.nan
        
    
    def clean_ids(self, df):
        
        """Function to clean df for none-integer data
        
        Args:
            df(object): the dataframe(pandas), which is the dataset
            
        Return:
            df(object): the cleaned data where 'id' was converted as 'int'
            
        """
        
        #Clean the ids of df
        df['id'] = df['id'].apply(self.clean_as_int)    
                                  
        #Filter all rows that have a null ID
        df = df[df['id'].notnull()]
        
        return df
    
    def main(self):
        
        """Function to return combineed with 'id' reference
        
        Args:
            none: 
        
        Return:
            combined dataframe (object):
        """
        self.df = self.clean_ids(self.df)
        self.cred_df = self.clean_ids(self.cred_df)
        self.key_df = self.clean_ids(self.key_df)
        
        # Merge keywords and credits into your main metadata dataframe
        self.df = self.df.merge(self.cred_df, on='id')
        self.df = self.df.merge(self.key_df, on='id')
        
        return self.df 
        
class CreateSoup:
    
    def __init__(self, cleaned_data):
        
        self.df = cleaned_data
        
    def get_native_obj(self, df):
        
        """Function to return combineed with 'id' reference
        
        Args:
            df(object): the dataframe(pandas), which is the dataset that contains 'features'
        
        Return:
            dataframe (object): the dataframe that applied 'literal_eval' function
        """
        from ast import literal_eval
        
        # Convert the stringified objects into the native python objects
        features = ['cast', 'crew', 'keywords', 'genres']
        
        for feature in features:
            df[feature] = df[feature].apply(literal_eval)
        
        return df
    
    def get_director(self, x):
        
        """Function to extract the director's name. If director is not listed, return NaN"""

        for crew_member in x:
            if crew_member['job'] == 'Director':
                return crew_member['name']
        
        return np.nan
    
    def generate_list(self, x, n=3):
        
        """Function to returns the list top 'n' elements or entire list"""
        
        if isinstance(x, list):
            
            names = [i['name'] for i in x]
            #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
            if len(names) > n:
                names = names[:n]
            return names

        #Return empty list in case of missing/malformed data
        
        return []

    def sanitize(self, x):
        
        """Function to sanitize data to prevent ambiguity. It removes spaces and converts to lowercase"""
        
        if isinstance(x, list):
            #Strip spaces and convert to lowercase
            return [str.lower(i.replace(" ", "")) for i in x]
        
        else:
            #Check if director exists. If not, return empty string
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''
    
    def create_soup(self, x):
        """Function that creates a soup out of the desired metadata"""
        
        return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

    def main(self):
        
        """Function to returns the soup of keywords, genres, cast, directer"""
        
        self.df = self.get_native_obj(self.df)
        
        #Define the new director feature
        self.df['director'] =  self.df['crew'].apply(self.get_director)
        
        #Apply the generate_list function to cast and keywords
        self.df['cast'] = self.df['cast'].apply(self.generate_list)
        self.df['keywords'] = self.df['keywords'].apply(self.generate_list)
        self.df['genres'] = self.df['genres'].apply(self.generate_list)
        
        #Only consider a maximum of 3 genres
        n=3
        self.df['genres'] = self.df['genres'].apply(lambda x: x[:n])

        #Apply the generate_list function to cast, keywords, director and genres
        for feature in ['cast', 'director', 'genres', 'keywords']:
            #print(feature)
            self.df[feature] = self.df[feature].apply(self.sanitize)
            
        # Create the new soup feature
        self.df['soup'] = self.df.apply(self.create_soup, axis=1)
        
        return self.df
    
class ContentBasedRecommender:
    
    def __init__(self, database_soup):
        
        self.df = database_soup
    
    def cal_tfidf(self, df, stop_words_list=['english']):
        
        """Function to creat the Term Frequency-Inverse Document Frequency (TF-IDF) matrix

        Args:
            df(object): the dataframe(pandas), which is the dataset that contain 'overview' documents of movies
            stop_words(list): the words that extremly commom in the 'overview' documents of movies
        
        Return:
            tfidf_matrix (tensor): the word vecterized-matrix
        """
        
        #Define a TF-IDF Vectorizer Object. Remove all english stopwords
        tfidf = TfidfVectorizer(stop_words=stop_words_list)

        #Replace NaN with an empty string
        df['soup'] = df['soup'].fillna('')

        #Construct the required TF-IDF matrix by applying the fit_transform method on the overview feature
        tfidf_matrix = tfidf.fit_transform(df['soup'])
        
        return tfidf_matrix
        
    def get_cosine_sim(self, tfidf_matrix):
        
        """Function to compute the cosine similarity matrix 

        Args:
            tfidf_matrix (tensor): the word vecterized-matrix
        
        Return:
            cosine similarity matrix(tensor)
        """
        
        # Compute the cosine similarity matrix
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        return cosine_sim
    
    def get_indices(self, df):
        
        """Function to construct a reverse mapping of indices and movie titles, 
        and drop duplicate titles(if any)"""
        
        indices = pd.Series(df.index, index=df['title']).drop_duplicates()
        
        return indices
        
    def main(self, title_input, see_top=25, random_seed=42):
        """Function to take in movie title as input and give recommendations using GPU.
        
        Args:
            title_input (string): The movie name.
            see_top (int): Number of recommendations to return.
            random_seed (int): Seed for reproducibility.

        Return:
            recommendation (pd.DataFrame): Recommended movies.
        """
        try:
            # Ensure the DataFrame index is sequential
            self.df = self.df.reset_index(drop=True)
            
            # Obtain the index of the movie that matches the title
            indices = self.get_indices(self.df)
            idx = indices.get(title_input)
            if idx is None:
                raise ValueError("Movie not found in database.")
            
            # Determine which text column to use for similarity calculations
            if 'description' in self.df.columns:
                text_col = 'description'
            elif 'overview' in self.df.columns:
                text_col = 'overview'
            else:
                raise ValueError("No suitable text column found (expected 'description' or 'overview').")
            
            # Calculate TF-IDF using the selected text column
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(self.df[text_col].values.astype('U'))
            
            # Compute cosine similarity for the target movie using sparse multiplication
            movie_vec = tfidf_matrix[idx]
            sim_scores = movie_vec.dot(tfidf_matrix.T)
            sim_scores = sim_scores.toarray().flatten()
            
            # Create a list of (movie_index, similarity_score) tuples
            sim_scores = list(enumerate(sim_scores))
            
            # Sort scores in descending order, skipping the first entry (the movie itself)
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:see_top+1]
            
            # Extract valid movie indices
            movie_indices = [i[0] for i in sim_scores]
            valid_indices = [i for i in movie_indices if 0 <= i < len(self.df)]
            
            # Get the recommended movies (selecting only 'title' and 'genres' columns)
            recommendations = self.df.iloc[valid_indices][['title', 'genres']]
            return recommendations.sample(n=min(len(recommendations), see_top), random_state=random_seed)

        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()




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
#meta_data
prepro = MergeCleanData(metadata_path, credits, keywords)
database = prepro.main()

make_soup= CreateSoup(database)
database_soup = make_soup.main()

recommender= ContentBasedRecommender(database_soup)

#knowledge
preference_scored= KnowledgeRecommender(metadata_path)
print("done")




@app.route('/recommend', methods=['POST'])
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

@app.route('/get_movies', methods=['POST'])
def get_movies():
    try:
        data = request.get_json()
        title_input = data.get('title_input')
        see_top = data.get('see_top', 10)
        
        if not title_input:
            return jsonify({"error": "Missing title_input."}), 400
        
        # Get movie recommendations based on title input
        recommended_movies = recommender.main(title_input=title_input, see_top=see_top)
        
        return jsonify(recommended_movies.to_dict('records'))
 
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/add_movie', methods=['POST'])
def add_movie():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        movie_name = data.get('movie_name')
        rating = data.get('rating')
        
        if not all([user_id, movie_name, rating]):
            return jsonify({"error": "Missing user_id, movie_name, or rating."}), 400

        # Call the function to add a new rating
        A1.add_new_rating(user_id, movie_name, rating)
        
        return jsonify({"message": "Movie rating added successfully!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/user_recommend', methods=['POST'])
def user_recommend():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        title = data.get('title')
        
        if not user_id or not title:
            return jsonify({"error": "Missing user_id or title."}), 400
        
        # Get recommendations using A1.main
        recommended_movies = A1.main(userId=user_id, title=title)
        
        return jsonify(recommended_movies.to_dict(orient='records'))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()