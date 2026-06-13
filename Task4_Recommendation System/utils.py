import os
import zipfile
import requests
import pandas as pd
import numpy as np

DATASET_DIR = "dataset"
ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP_PATH = os.path.join(DATASET_DIR, "ml-latest-small.zip")

def download_and_prepare_dataset(force=False):
    """
    Downloads MovieLens Latest Small dataset and extracts it.
    If it fails, it generates a synthetic fallback dataset.
    """
    os.makedirs(DATASET_DIR, exist_ok=True)
    movies_path = os.path.join(DATASET_DIR, "movies.csv")
    ratings_path = os.path.join(DATASET_DIR, "ratings.csv")
    tags_path = os.path.join(DATASET_DIR, "tags.csv")
    
    # Check if files already exist
    if not force and os.path.exists(movies_path) and os.path.exists(ratings_path):
        print("Dataset files already exist. Skipping download.")
        return movies_path, ratings_path, tags_path

    print("Attempting to download MovieLens dataset...")
    try:
        response = requests.get(ZIP_URL, timeout=15)
        response.raise_for_status()
        
        with open(ZIP_PATH, 'wb') as f:
            f.write(response.content)
            
        print("Download complete. Extracting files...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            # MovieLens zip contains a folder ml-latest-small/
            zip_ref.extractall(DATASET_DIR)
            
        # Move files to dataset/ root for a flatter structure
        extracted_dir = os.path.join(DATASET_DIR, "ml-latest-small")
        for file_name in ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]:
            src = os.path.join(extracted_dir, file_name)
            dst = os.path.join(DATASET_DIR, file_name)
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)
                
        # Clean up zip and temporary directory
        os.remove(ZIP_PATH)
        import shutil
        if os.path.exists(extracted_dir):
            shutil.rmtree(extracted_dir)
            
        print("MovieLens dataset prepared successfully.")
        return movies_path, ratings_path, tags_path
        
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        print("Generating realistic synthetic dataset as fallback...")
        return generate_synthetic_dataset()

def generate_synthetic_dataset():
    """
    Generates a rich, realistic movie recommendation dataset.
    Includes genres, user tastes, ratings, and tags.
    """
    os.makedirs(DATASET_DIR, exist_ok=True)
    movies_path = os.path.join(DATASET_DIR, "movies.csv")
    ratings_path = os.path.join(DATASET_DIR, "ratings.csv")
    tags_path = os.path.join(DATASET_DIR, "tags.csv")

    # 1. Define synthetic movies
    movies_data = [
        # Sci-Fi / Action
        (1, "Inception", "Action|Sci-Fi|Thriller", "dreams mind-bending action leonardo physics"),
        (2, "Interstellar", "Sci-Fi|Drama|IMAX", "space time-travel astrophysics cornfield emotional"),
        (3, "The Matrix", "Action|Sci-Fi", "simulation cyberpunk red-pill martial-arts"),
        (4, "Star Wars: Episode IV - A New Hope", "Action|Adventure|Sci-Fi", "space-opera force lightsaber classic"),
        (5, "Blade Runner 2049", "Sci-Fi|Thriller", "cyberpunk neon replicant atmospheric slow-burn"),
        (6, "The Martian", "Adventure|Drama|Sci-Fi", "mars survival space science comedy"),
        (7, "Arrival", "Sci-Fi|Drama|Mystery", "aliens linguistics time emotional smart"),
        (8, "Edge of Tomorrow", "Action|Sci-Fi", "time-loop aliens exoskeleton action-packed"),
        # Drama / Classics
        (9, "The Shawshank Redemption", "Drama", "prison hope friendship classic masterpieces"),
        (10, "The Godfather", "Crime|Drama", "mafia family crime classic pacino"),
        (11, "Pulp Fiction", "Comedy|Crime|Drama|Thriller", "tarantino dialogues nonlinear crime cool"),
        (12, "Forrest Gump", "Comedy|Drama|Romance", "life history emotional sweet Tom-Hanks"),
        (13, "Fight Club", "Action|Crime|Drama|Thriller", "anarchy split-personality twist psychological"),
        (14, "The Green Mile", "Crime|Drama|Fantasy", "prison supernatural emotional stephen-king"),
        (15, "GoodFellas", "Crime|Drama", "mafia scorsese crime true-story biography"),
        # Romance / Comedy
        (16, "Titanic", "Drama|Romance", "shipwreck disaster love emotional leo-di-caprio"),
        (17, "The Notebook", "Drama|Romance", "love-story tears relationship notebook classic-romance"),
        (18, "La La Land", "Comedy|Drama|Musical|Romance", "jazz musical hollywood love dream-chasing"),
        (19, "When Harry Met Sally...", "Comedy|Romance", "friends-to-lovers relationships dialogue classic-comedy"),
        (20, "About Time", "Comedy|Drama|Fantasy|Romance", "time-travel family love sweet British"),
        (21, "Pride & Prejudice", "Drama|Romance", "period-piece literature class jane-austen"),
        (22, "500 Days of Summer", "Comedy|Drama|Romance", "indie relationship heartbreak music"),
        # Animation / Kids
        (23, "Toy Story", "Adventure|Animation|Children|Comedy|Fantasy", "toys friendship pixar classic childhood"),
        (24, "Spirited Away", "Adventure|Animation|Fantasy", "anime ghibli spirits masterpiece magical"),
        (25, "Finding Nemo", "Adventure|Animation|Children|Comedy", "fish ocean family pixar adventure"),
        (26, "The Lion King", "Adventure|Animation|Children|Drama|Musical", "disney safari music tragedy childhood"),
        (27, "Monsters, Inc.", "Adventure|Animation|Children|Comedy|Fantasy", "monsters pixar friendship funny"),
        (28, "WALL-E", "Adventure|Animation|Children|Romance|Sci-Fi", "robot space post-apocalyptic silent pixar"),
        (29, "How to Train Your Dragon", "Adventure|Animation|Children|Fantasy", "dragons friendship adventure viking"),
        # Horror / Thriller
        (30, "The Silence of the Lambs", "Crime|Horror|Thriller", "psychopath fbi serial-killer cannibal suspense"),
        (31, "The Shining", "Horror", "hotel isolation madness jack-nicholson stephen-king"),
        (32, "Psycho", "Horror|Mystery|Thriller", "hitchcock motel classic murder suspense"),
        (33, "Get Out", "Horror|Mystery|Thriller", "social-commentary suspense twist modern-horror"),
        (34, "Se7en", "Mystery|Thriller", "serial-killer rain dark fbi twist fincher"),
        (35, "Shutter Island", "Drama|Mystery|Thriller", "island asylum twist leonardo psychological"),
        (36, "A Quiet Place", "Horror|Sci-Fi|Thriller", "silent monsters survival family suspense"),
    ]
    
    movies_df = pd.DataFrame(movies_data, columns=["movieId", "title", "genres", "raw_tags"])
    movies_df[["movieId", "title", "genres"]].to_csv(movies_path, index=False)
    
    # 2. Generate ratings with user profiles
    # We will generate ratings for 50 users.
    # Group 1 (Users 1-15): Sci-Fi/Action fans. Rate Sci-Fi high (4-5), Romance/Drama low (1-3).
    # Group 2 (Users 16-30): Romance/Drama/Comedy fans. Rate Romance/Drama high (4-5), Sci-Fi/Horror low (1-3).
    # Group 3 (Users 31-40): Animation/Family fans. Rate Animation high (4-5), Horror low (1-2).
    # Group 4 (Users 41-50): General/Casual fans. Rate everything randomly (2-5).
    
    np.random.seed(42)
    ratings = []
    tags = []
    
    for user_id in range(1, 51):
        if user_id <= 15:
            # Sci-Fi / Action fans
            preferred_genres = ["Sci-Fi", "Action"]
            disliked_genres = ["Romance", "Musical"]
        elif user_id <= 30:
            # Romance / Drama / Comedy
            preferred_genres = ["Romance", "Drama"]
            disliked_genres = ["Sci-Fi", "Horror"]
        elif user_id <= 40:
            # Animation
            preferred_genres = ["Animation", "Children"]
            disliked_genres = ["Horror", "Thriller"]
        else:
            # Casual
            preferred_genres = []
            disliked_genres = []
            
        # Each user rates a random 15-25 movies out of the 36 available
        num_rated = np.random.randint(15, 26)
        rated_movies = np.random.choice(movies_df["movieId"].values, size=num_rated, replace=False)
        
        for movie_id in rated_movies:
            movie_row = movies_df[movies_df["movieId"] == movie_id].iloc[0]
            movie_genres = movie_row["genres"].split("|")
            
            # Determine rating based on user preference
            rating = 3.0  # Base
            if any(g in preferred_genres for g in movie_genres):
                rating += np.random.uniform(1.0, 2.0)
            elif any(g in disliked_genres for g in movie_genres):
                rating -= np.random.uniform(1.0, 2.0)
            else:
                rating += np.random.uniform(-1.0, 1.0)
                
            rating = np.clip(np.round(rating * 2) / 2.0, 0.5, 5.0)  # Round to nearest 0.5, clip to [0.5, 5.0]
            timestamp = 1262304000 + np.random.randint(0, 315360000) # random time between 2010 and 2020
            
            ratings.append([user_id, movie_id, rating, timestamp])
            
            # Occasionally add a user tag
            if np.random.rand() < 0.3:
                raw_tag_list = movie_row["raw_tags"].split()
                selected_tag = np.random.choice(raw_tag_list)
                tags.append([user_id, movie_id, selected_tag, timestamp])
                
    ratings_df = pd.DataFrame(ratings, columns=["userId", "movieId", "rating", "timestamp"])
    ratings_df.to_csv(ratings_path, index=False)
    
    tags_df = pd.DataFrame(tags, columns=["userId", "movieId", "tag", "timestamp"])
    tags_df.to_csv(tags_path, index=False)
    
    print("Synthetic fallback dataset generated.")
    return movies_path, ratings_path, tags_path

# Evaluation Metrics

def rmse(y_true, y_pred):
    """Calculates Root Mean Squared Error."""
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def mae(y_true, y_pred):
    """Calculates Mean Absolute Error."""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def precision_recall_at_k(recommendations_dict, actual_likes_dict, k=5):
    """
    Computes Precision@K and Recall@K for a set of recommendations and actual likes.
    Args:
        recommendations_dict: dict of {user_id: list of recommended movie_ids}
        actual_likes_dict: dict of {user_id: set of actual liked movie_ids in test set}
        k: number of recommendations to evaluate
    Returns:
        mean_precision: average precision@K across users
        mean_recall: average recall@K across users
    """
    precisions = []
    recalls = []
    
    for user_id, actual_likes in actual_likes_dict.items():
        if not actual_likes:
            continue
            
        user_recs = recommendations_dict.get(user_id, [])[:k]
        if not user_recs:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
            
        # Number of recommended items that are actually in user's liked list
        n_rel_and_rec = len(set(user_recs).intersection(actual_likes))
        
        # Precision@K = (Relevant & Recommended) / Recommended
        precisions.append(n_rel_and_rec / len(user_recs))
        
        # Recall@K = (Relevant & Recommended) / Relevant
        recalls.append(n_rel_and_rec / len(actual_likes))
        
    mean_precision = np.mean(precisions) if precisions else 0.0
    mean_recall = np.mean(recalls) if recalls else 0.0
    
    return mean_precision, mean_recall

def split_ratings_train_test(ratings_df, test_ratio=0.2, random_state=42):
    """
    Splits rating interactions into train and test sets.
    Ensures that for each user, we try to keep their ratings split proportionally.
    """
    np.random.seed(random_state)
    test_indices = []
    
    # Group by user and take test_ratio of ratings for test
    for _, group in ratings_df.groupby('userId'):
        n_ratings = len(group)
        if n_ratings >= 5:
            n_test = int(np.ceil(n_ratings * test_ratio))
            indices = np.random.choice(group.index, size=n_test, replace=False)
            test_indices.extend(indices)
            
    test_df = ratings_df.loc[test_indices]
    train_df = ratings_df.drop(test_indices)
    
    return train_df, test_df
