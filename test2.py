import pandas as pd
import numpy as np
import time
import warnings
import re 

# --- CPU Libraries ---
from sklearn.feature_extraction.text import TfidfVectorizer 
# We still need sklearn's cosine_similarity as a fallback
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity 
from fuzzywuzzy import fuzz
from surprise import SVD 
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import scipy.sparse # To work with sklearn's sparse output

# --- GPU Libraries (Optional - CuPy) ---
try:
    import cupy as cp
    import cupyx.scipy.sparse as cupy_sparse # For sparse matrix operations on GPU
    CUPY_AVAILABLE = True
    print("CuPy found. GPU acceleration for Cosine Similarity enabled.")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not found or CUDA not configured correctly. Falling back to CPU for all computations.")
    warnings.warn("CuPy not available. Cosine Similarity will run on CPU.")

# --- Configuration ---
USE_GPU = CUPY_AVAILABLE # Controls GPU usage attempt
N_RECOMMENDATIONS = 10
MY_FAVORITE_MOVIES = ['Iron Man'] 

# --- Download NLTK data ---
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("Downloading NLTK stopwords data...")
    nltk.download('stopwords')
    print("Download complete.")

# --- Load Data ---
print("Loading data...")
start_time = time.time()
movies_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv("ratings_small.csv")
print(f"Data loaded in {time.time() - start_time:.2f} seconds.")

# --- Preprocessing ---
print("Preprocessing data...")
start_time = time.time()

ratings_array = ratings_df['rating'].unique()
ratings_array = np.sort(ratings_array) # Sort for consistent get_rating_from_prediction
max_rating = np.max(ratings_array)
min_rating = np.min(ratings_array)

# Movie maps
movie_map = pd.Series(movies_df.movieId.values, index=movies_df.title).to_dict()
reverse_movie_map = {v: k for k, v in movie_map.items()}
movieId_to_index_map = pd.Series(movies_df.index.values, index=movies_df.movieId).to_dict()
index_to_movieId_map = {v: k for k, v in movieId_to_index_map.items()}
index_to_title_map = movies_df['title'].to_dict() 

# Use sets for faster lookups
all_movie_ids_set = set(movies_df['movieId'].unique())

# --- Helper Functions ---

def get_movieId(movie_name, movie_map_dict):
    # (Function remains the same as previous CPU version)
    if movie_name in movie_map_dict:
        return movie_map_dict[movie_name]
    best_match = None
    highest_ratio = -1
    for title, movie_id in movie_map_dict.items():
        ratio = fuzz.ratio(title.lower(), movie_name.lower())
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = (title, movie_id, ratio)
    if best_match and best_match[2] >= 60:
         print(f"Movie '{movie_name}' not found exactly. Using best match: '{best_match[0]}' (Ratio: {best_match[2]}%)")
         return best_match[1]
    else:
        print(f"Warning: Movie '{movie_name}' could not be found in the database.")
        return None

_stopwords = set(stopwords.words('english'))
_porter_stemmer = PorterStemmer()

def tokenizer(text):
    # (Function remains the same as previous CPU version)
    if not isinstance(text, str): return []
    tokens = [ _porter_stemmer.stem(word.lower()) for word in text.split('|') if word and word.lower() not in _stopwords ]
    return tokens
    
def get_rating_from_prediction(prediction, sorted_ratings_array):
   # (Function remains the same as previous CPU version)
    idx = np.searchsorted(sorted_ratings_array, prediction, side="left")
    if idx == 0: return sorted_ratings_array[0]
    if idx == len(sorted_ratings_array): return sorted_ratings_array[-1]
    left_neighbor = sorted_ratings_array[idx - 1]
    right_neighbor = sorted_ratings_array[idx]
    if (prediction - left_neighbor) < (right_neighbor - prediction): return left_neighbor
    else: return right_neighbor

# --- CuPy Cosine Similarity Helper ---
def cupy_cosine_similarity_sparse(sparse_matrix_gpu):
    """Calculates cosine similarity for a CuPy sparse matrix."""
    # Normalize rows (vectors) to unit length
    # norm = cp.sqrt(sparse_matrix_gpu.multiply(sparse_matrix_gpu).sum(axis=1)) # Element-wise square then sum
    norm = cupy_sparse.linalg.norm(sparse_matrix_gpu, axis=1)
    # Handle potential division by zero for rows with zero norm (e.g., no genres)
    norm = cp.where(norm == 0, 1.0, norm) # Avoid division by zero, similarity will be 0 anyway
    
    # Perform normalization using broadcasted division
    # Convert norm to column vector for broadcasting
    norm_col = norm.reshape(-1, 1) 
    normalized_matrix = sparse_matrix_gpu / norm_col 
    
    # Calculate cosine similarity via matrix multiplication
    # Result might be dense or sparse depending on CuPy version/optimizations
    # Ensure it's compatible with downstream use (e.g., convert to dense if needed)
    similarity_matrix = normalized_matrix @ normalized_matrix.T 
    
    # Convert to dense if it's not already, as argsort works on dense
    if cupy_sparse.issparse(similarity_matrix):
         # Experimental: Check if to_dense() is feasible memory-wise
         print("CuPy similarity matrix is sparse, converting to dense for argsort (may require significant GPU memory)...")
         try:
            similarity_matrix = similarity_matrix.toarray() # .toarray() converts to dense CuPy array
         except MemoryError:
             print("Error: Insufficient GPU memory to convert sparse similarity matrix to dense.")
             print("Falling back to CPU cosine similarity calculation.")
             return None # Signal failure to fallback
         except AttributeError: # Maybe toarray() not available? Use get() -> numpy -> cupy
              print("Warning: .toarray() not found, attempting alternative dense conversion.")
              try:
                similarity_matrix = cp.asarray(similarity_matrix.get())
              except MemoryError:
                  print("Error: Insufficient memory for alternative dense conversion.")
                  return None
    
    return similarity_matrix


print(f"Preprocessing done in {time.time() - start_time:.2f} seconds.")

# --- Item-Based Recommendation Setup ---
print("Setting up Item-Based Recommendation...")
start_time = time.time()

# Prepare genres
movies_df['genres_cleaned'] = movies_df['genres'].fillna('').astype(str) 

# --- Step 1: TF-IDF on CPU ---
print("Calculating TF-IDF matrix (CPU: scikit-learn)...")
tfid_vectorizer = TfidfVectorizer(tokenizer=tokenizer, token_pattern=None) 
tfidf_matrix_cpu = tfid_vectorizer.fit_transform(movies_df['genres_cleaned'])
print("TF-IDF matrix shape (CPU):", tfidf_matrix_cpu.shape)
print(f"TF-IDF calculation took {time.time() - start_time:.2f} seconds.")

# --- Step 2: Calculate Cosine Similarity (GPU or CPU) ---
cos_sim_matrix = None 
tfidf_matrix_gpu = None # Keep track if GPU matrix exists

if USE_GPU:
    print("Attempting Cosine Similarity calculation on GPU (CuPy)...")
    transfer_start_time = time.time()
    try:
        # --- Step 2a: Transfer TF-IDF matrix to GPU ---
        # Ensure it's in CSR format for CuPy compatibility
        if not isinstance(tfidf_matrix_cpu, scipy.sparse.csr_matrix):
            tfidf_matrix_cpu = tfidf_matrix_cpu.tocsr() 
            
        tfidf_matrix_gpu = cupy_sparse.csr_matrix(tfidf_matrix_cpu)
        print(f"Transferred TF-IDF matrix to GPU in {time.time() - transfer_start_time:.2f} seconds.")
        
        # --- Step 2b: Calculate Cosine Similarity on GPU ---
        calc_start_time = time.time()
        cos_sim_matrix_gpu = cupy_cosine_similarity_sparse(tfidf_matrix_gpu)
        
        if cos_sim_matrix_gpu is not None:
            cos_sim_matrix = cos_sim_matrix_gpu # Assign the GPU matrix
            print(f"Calculated Cosine Similarity on GPU in {time.time() - calc_start_time:.2f} seconds.")
            print("Cosine Similarity matrix shape (GPU):", cos_sim_matrix.shape)
        else:
            # Fallback signaled from cupy_cosine_similarity_sparse (e.g., MemoryError)
            USE_GPU = False # Force CPU path going forward
            del tfidf_matrix_gpu # Clean up GPU memory if transfer succeeded but calc failed
            tfidf_matrix_gpu = None 
            
    except Exception as e:
        print(f"Error during GPU setup or calculation: {e}")
        print("Falling back to CPU Cosine Similarity calculation.")
        USE_GPU = False # Force CPU path
        del tfidf_matrix_gpu # Clean up GPU memory if transfer succeeded but calc failed
        tfidf_matrix_gpu = None 

# --- Step 2c: Fallback to CPU if GPU failed or wasn't requested ---
if cos_sim_matrix is None:
    print("Calculating Cosine Similarity matrix (CPU: scikit-learn)...")
    calc_start_time = time.time()
    # sklearn handles sparse input directly
    cos_sim_matrix = sklearn_cosine_similarity(tfidf_matrix_cpu, dense_output=True) 
    print(f"Calculated Cosine Similarity on CPU in {time.time() - calc_start_time:.2f} seconds.")
    print("Cosine Similarity matrix shape (CPU):", cos_sim_matrix.shape)

# Clean up the large CPU TF-IDF matrix if we successfully moved to GPU
if tfidf_matrix_gpu is not None:
     del tfidf_matrix_cpu

print(f"Item-Based setup done in {time.time() - start_time:.2f} seconds.")


# --- User-Based Recommendation Setup (Surprise SVD - CPU Bound) ---
# (This section remains unchanged from the previous CPU-only version)
print("Setting up User-Based Recommendation (Surprise SVD - CPU)...")
start_time = time.time()
features = ['userId', 'movieId', 'rating']
reader = Reader(rating_scale=(min_rating, max_rating))
data = Dataset.load_from_df(ratings_df[features], reader)
param_grid = {'n_epochs': [10, 20], 'lr_all': [0.005, 0.01], 'reg_all': [0.02, 0.1]} 
print("Running GridSearchCV for SVD...")
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1) 
gs.fit(data)
print("Best SVD parameters (RMSE):", gs.best_params['rmse'])
best_params = gs.best_params['rmse']
print("Training final SVD model...")
trainset = data.build_full_trainset()
model_svd = SVD(**best_params)
model_svd.fit(trainset)
print(f"User-Based setup (SVD training) done in {time.time() - start_time:.2f} seconds.")


# --- Recommendation Functions ---

def make_recommendation_item_based(
    similarity_matrix, # Can be NumPy array or CuPy array
    target_movie_index,
    n_recommendations, 
    index_to_title_map,
    watched_movie_indices_set):
    """
    Generates item-based recommendations using precomputed similarity.
    Uses CuPy for sorting if similarity_matrix is on GPU.
    """
    
    is_gpu = CUPY_AVAILABLE and isinstance(similarity_matrix, cp.ndarray)
    
    if target_movie_index is None or target_movie_index >= similarity_matrix.shape[0]:
        print(f"Error: Invalid target_movie_index ({target_movie_index}).")
        return []
        
    try:
        if is_gpu:
            # --- GPU Path ---
            # Get similarity scores for the target movie index (CuPy array)
            # Ensure the matrix is dense for simple indexing
            if cupy_sparse.issparse(similarity_matrix):
                 # This shouldn't happen if cupy_cosine_similarity_sparse worked
                 print("Warning: GPU similarity matrix is sparse unexpectedly. Converting.")
                 sim_scores_gpu = similarity_matrix[target_movie_index].toarray().flatten()
            else:    
                sim_scores_gpu = similarity_matrix[target_movie_index]
                
            # Sort indices based on scores (descending) on GPU
            # argsort gives indices that *would* sort the array
            sorted_indices_gpu = cp.argsort(sim_scores_gpu)[::-1]
            
            # Transfer only the necessary sorted indices back to CPU
            # Transferring top N + buffer is usually enough, but transfer all for simplicity here
            sorted_indices_cpu = cp.asnumpy(sorted_indices_gpu)
            # Clean up GPU memory associated with this function call if needed
            # del sim_scores_gpu, sorted_indices_gpu 
            
        else:
            # --- CPU Path ---
            sim_scores_cpu = similarity_matrix[target_movie_index]
            sorted_indices_cpu = np.argsort(sim_scores_cpu)[::-1]

        # Filter and collect recommendations (on CPU using NumPy indices)
        recommendations = []
        count = 0
        for idx in sorted_indices_cpu:
            # Convert NumPy int64 index to standard Python int for set lookup and dict key
            idx_int = int(idx) 
            if idx_int != target_movie_index and idx_int not in watched_movie_indices_set and idx_int in index_to_title_map:
                recommendations.append(index_to_title_map[idx_int]) 
                count += 1
                if count >= n_recommendations:
                    break
                    
        return recommendations

    except Exception as e:
        print(f"Error during item-based recommendation generation: {e}")
        # Potentially handle GPU memory errors here more gracefully if they occur
        return []


def make_recommendation_user_based(
    svd_model, 
    all_movie_ids_set, 
    user_id,
    watched_movie_ids_set,
    n_recommendations, 
    id_to_movie_map,
    sorted_ratings_array):
    # (Function remains the same as previous CPU-only version)
    movies_to_predict = all_movie_ids_set - watched_movie_ids_set 
    if not movies_to_predict:
        print(f"User {user_id} has potentially rated all available movies!")
        return []
    print(f"Predicting ratings for {len(movies_to_predict)} unwatched movies for user {user_id}...")
    predictions = []
    for movie_id in movies_to_predict:
        pred = svd_model.predict(user_id, movie_id)
        predictions.append(pred)
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n_recs = []
    for pred in predictions[:n_recommendations]:
        movie_title = id_to_movie_map.get(pred.iid, "Unknown Title")
        top_n_recs.append(movie_title)
    return top_n_recs

# --- Generate Recommendations ---

# Assume a user ID
USER_ID_TO_RECOMMEND = 1 
watched_movies_user = set(ratings_df[ratings_df['userId'] == USER_ID_TO_RECOMMEND]['movieId'].unique())

# Process favorite movies
fav_movie_ids = []
fav_movie_indices = []
for movie_name in MY_FAVORITE_MOVIES:
    movie_id = get_movieId(movie_name, movie_map)
    if movie_id:
        fav_movie_ids.append(movie_id)
        movie_index = movieId_to_index_map.get(movie_id)
        if movie_index is not None:
             fav_movie_indices.append(movie_index)
        else:
             print(f"Warning: Could not find index for movieId {movie_id} ({movie_name})")
             
combined_watched_ids = watched_movies_user.union(set(fav_movie_ids))
combined_watched_indices = set(movieId_to_index_map.get(mid) for mid in combined_watched_ids if mid in movieId_to_index_map)


# --- Item-Based Call ---
print("\n--- Generating Item-Based Recommendations ---")
print(f"Using {'GPU (CuPy)' if USE_GPU and isinstance(cos_sim_matrix, cp.ndarray) else 'CPU (NumPy/Sklearn)'} for similarity sorting.")
start_time = time.time()
recommends_item_based = []
if fav_movie_indices:
    target_movie_index = fav_movie_indices[0] 
    recommends_item_based = make_recommendation_item_based(
        similarity_matrix=cos_sim_matrix, # Pass the potentially CuPy matrix
        target_movie_index=target_movie_index,
        n_recommendations=N_RECOMMENDATIONS,
        index_to_title_map=index_to_title_map,
        watched_movie_indices_set=combined_watched_indices
    )
else:
    print("No valid favorite movies found to seed item-based recommendations.")

print(f"Item-Based recommendations generated in {time.time() - start_time:.4f} seconds.")
print("\n------------- Item-Based (Content Similarity) Recommendations -------------")
print(f'Movies similar to "{MY_FAVORITE_MOVIES[0]}" (excluding watched):')
if recommends_item_based:
    for i, title in enumerate(recommends_item_based):
        print(f"{i+1}. {title}")
else:
    print("Could not generate item-based recommendations.")

# --- User-Based Call (Remains CPU) ---
print("\n--- Generating User-Based (SVD - CPU) Recommendations ---")
start_time = time.time()
recommends_user_based = make_recommendation_user_based(
    svd_model=model_svd,
    all_movie_ids=all_movie_ids_set,
    user_id=USER_ID_TO_RECOMMEND,
    watched_movie_ids_set=combined_watched_ids, 
    n_recommendations=N_RECOMMENDATIONS,
    id_to_movie_map=reverse_movie_map,
    sorted_ratings_array=ratings_array 
)

print(f"User-Based recommendations generated in {time.time() - start_time:.4f} seconds.")
print("\n------------- User-Based (Collaborative Filtering) Recommendations -------------")
print(f'Top recommendations for User {USER_ID_TO_RECOMMEND} (excluding watched/favorites):')
if recommends_user_based:
    for i, title in enumerate(recommends_user_based):
        print(f"{i+1}. {title}")
else:
    print("Could not generate user-based recommendations.")

# --- Optional: Clean up GPU memory explicitly ---
if CUPY_AVAILABLE:
    try:
        # Remove large matrices from GPU memory if they exist
        if 'cos_sim_matrix' in locals() and isinstance(cos_sim_matrix, cp.ndarray):
            del cos_sim_matrix
        if 'tfidf_matrix_gpu' in locals() and tfidf_matrix_gpu is not None:
             del tfidf_matrix_gpu
        # Clear CuPy memory pool
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        print("\nCuPy GPU memory cleared.")
    except NameError:
        pass # Variables might not exist if GPU path wasn't taken
    except Exception as e:
        print(f"\nError during CuPy cleanup: {e}")


print("\nDone.")