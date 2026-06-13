import os
import pandas as pd
import numpy as np
import pickle
from content_based import ContentBasedRecommender
from collaborative_filtering import CollaborativeFilteringRecommender

class HybridRecommender:
    def __init__(self, k_neighbors=40):
        self.content_model = ContentBasedRecommender()
        # We hold two CF models: one User-based and one Item-based, and can toggle between them.
        self.cf_user_model = CollaborativeFilteringRecommender(kind='user', k_neighbors=k_neighbors)
        self.cf_item_model = CollaborativeFilteringRecommender(kind='item', k_neighbors=k_neighbors)
        self.movies_df = None
        self.ratings_df = None
        self.global_mean = 3.5

    def fit(self, movies_df, ratings_df, tags_df=None):
        """
        Fits both content-based and collaborative filtering models.
        """
        self.movies_df = movies_df.copy()
        self.ratings_df = ratings_df.copy()
        self.global_mean = ratings_df['rating'].mean()

        # Fit content-based model
        self.content_model.fit(movies_df, tags_df)

        # Fit collaborative filtering models
        self.cf_user_model.fit(ratings_df)
        self.cf_item_model.fit(ratings_df)
        print("Hybrid Recommender fitted successfully.")

    def get_popular_movies(self, top_n=10, min_ratings=5):
        """
        Calculates movie popularity using Bayesian Average rating.
        Bayesian Average = (v * R + m * C) / (v + m)
        where:
        v = number of ratings for the movie
        R = average rating of the movie
        m = prior weight (minimum ratings threshold)
        C = mean rating across all movies
        """
        if self.ratings_df is None or self.movies_df is None:
            return pd.DataFrame()

        # Count ratings and mean rating for each movie
        stats = self.ratings_df.groupby('movieId')['rating'].agg(['count', 'mean']).reset_index()
        stats.rename(columns={'count': 'vote_count', 'mean': 'vote_average'}, inplace=True)

        C = self.global_mean
        m = min_ratings

        # Apply Bayesian Average formula
        stats['bayesian_rating'] = (stats['vote_count'] * stats['vote_average'] + m * C) / (stats['vote_count'] + m)

        # Merge with movies to get titles and genres
        popular_df = stats.merge(self.movies_df, on='movieId')
        popular_df = popular_df.sort_values(by='bayesian_rating', ascending=False)

        return popular_df.head(top_n)

    def get_cold_start_recommendations(self, favorite_genres, top_n=5, min_ratings=3):
        """
        Recommends popular movies matching the user's selected genres (cold-start).
        favorite_genres: list of strings (e.g. ['Action', 'Sci-Fi'])
        """
        popular_df = self.get_popular_movies(top_n=200, min_ratings=min_ratings)
        
        if popular_df.empty:
            # Fallback to simple movie list if ratings are empty
            filtered_movies = self.movies_df.copy()
        else:
            filtered_movies = popular_df
            
        if favorite_genres:
            # Filter movies that have at least one of the selected genres
            # Lowercase genres for safety
            fav_genres_lower = [g.lower().strip() for g in favorite_genres]
            
            def matches_genres(genre_str):
                movie_genres = [g.lower().strip() for g in str(genre_str).split('|')]
                return any(g in fav_genres_lower for g in movie_genres)
                
            filtered_movies = filtered_movies[filtered_movies['genres'].apply(matches_genres)]
            
        return filtered_movies.head(top_n)

    def predict_rating(self, user_id, movie_id, user_ratings_dict=None, cf_kind='item', alpha=0.5):
        """
        Predicts rating using a hybrid blend:
        Rating = alpha * CF_Rating + (1 - alpha) * Content_Rating
        """
        # Get Collaborative Filtering rating prediction
        cf_model = self.cf_item_model if cf_kind == 'item' else self.cf_user_model
        cf_pred = cf_model.predict_rating(user_id, movie_id, user_ratings_dict)

        # Get Content-Based rating prediction
        # Since Content-Based expects user_ratings dict, we extract it if user_id is given
        if user_ratings_dict is None and self.ratings_df is not None:
            # Find rating history of target user
            history = self.ratings_df[self.ratings_df['userId'] == user_id]
            user_ratings = dict(zip(history['movieId'], history['rating']))
        else:
            user_ratings = user_ratings_dict or {}

        cb_pred = self.content_model.predict_rating(user_ratings, movie_id)

        # Combine predictions
        hybrid_pred = alpha * cf_pred + (1.0 - alpha) * cb_pred
        return np.clip(hybrid_pred, 0.5, 5.0)

    def recommend(self, user_id, top_n=5, user_ratings_dict=None, cf_kind='item', alpha=0.5, exclude_rated=True):
        """
        Generates recommendations for user_id based on hybrid scores.
        """
        # Determine user rating history to exclude already rated items
        if user_ratings_dict is not None:
            user_ratings = user_ratings_dict
        elif self.ratings_df is not None:
            history = self.ratings_df[self.ratings_df['userId'] == user_id]
            user_ratings = dict(zip(history['movieId'], history['rating']))
        else:
            user_ratings = {}

        rated_movies = set(user_ratings.keys()) if exclude_rated else set()

        # Cold Start Check: If user has less than 3 ratings and we have favorite genres, we recommend popular items.
        # But if we are running standard evaluation, we just run the predictions.
        if len(user_ratings) < 3 and user_ratings_dict is None:
            # Let the UI handle cold start or return popular movies
            popular = self.get_popular_movies(top_n=top_n)
            return list(zip(popular['movieId'].values, popular['bayesian_rating'].values))

        predictions = []
        all_movies = self.movies_df['movieId'].values

        for m_id in all_movies:
            if m_id in rated_movies:
                continue
            pred_score = self.predict_rating(user_id, m_id, user_ratings, cf_kind, alpha)
            predictions.append((m_id, pred_score))

        # Sort by hybrid prediction score descending
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_n]

    def save(self, dir_path):
        """Saves all underlying models to a folder."""
        os.makedirs(dir_path, exist_ok=True)
        self.content_model.save(os.path.join(dir_path, "content_model.pkl"))
        self.cf_user_model.save(os.path.join(dir_path, "cf_user_model.pkl"))
        self.cf_item_model.save(os.path.join(dir_path, "cf_item_model.pkl"))
        
        # Save metadata
        metadata = {
            'global_mean': self.global_mean,
            'movies_df': self.movies_df,
            'ratings_df': self.ratings_df
        }
        with open(os.path.join(dir_path, "metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Hybrid Recommender files saved to {dir_path}")

    def load(self, dir_path):
        """Loads all underlying models from a folder."""
        self.content_model.load(os.path.join(dir_path, "content_model.pkl"))
        self.cf_user_model.load(os.path.join(dir_path, "cf_user_model.pkl"))
        self.cf_item_model.load(os.path.join(dir_path, "cf_item_model.pkl"))
        
        # Load metadata
        with open(os.path.join(dir_path, "metadata.pkl"), 'rb') as f:
            metadata = pickle.load(f)
            self.global_mean = metadata['global_mean']
            self.movies_df = metadata['movies_df']
            self.ratings_df = metadata['ratings_df']
        print(f"Hybrid Recommender loaded from {dir_path}")
