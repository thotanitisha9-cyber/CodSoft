import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFilteringRecommender:
    def __init__(self, kind='user', k_neighbors=40):
        """
        kind: 'user' for User-Based Collaborative Filtering,
              'item' for Item-Based Collaborative Filtering.
        k_neighbors: number of similar users/items to consider for prediction.
        """
        self.kind = kind
        self.k_neighbors = k_neighbors
        self.user_item = None
        self.user_means = None
        self.user_item_centered = None
        self.similarity_matrix = None
        self.global_mean = 3.5
        
    def fit(self, ratings_df):
        """
        Pivots ratings into a user-item matrix and computes the similarity matrix.
        """
        print(f"Fitting {self.kind.capitalize()}-Based Collaborative Filtering model...")
        self.global_mean = ratings_df['rating'].mean()
        
        # 1. Pivot ratings to user-item matrix
        self.user_item = ratings_df.pivot(index='userId', columns='movieId', values='rating')
        
        # Calculate user means (ignoring NaN)
        self.user_means = self.user_item.mean(axis=1)
        
        # Center the user ratings (subtract mean rating of each user to handle bias)
        self.user_item_centered = self.user_item.sub(self.user_means, axis=0).fillna(0)
        
        # 2. Compute similarities
        if self.kind == 'user':
            # Cosine similarity between user rating vectors (rows)
            sim = cosine_similarity(self.user_item_centered.values)
            self.similarity_matrix = pd.DataFrame(sim, index=self.user_item.index, columns=self.user_item.index)
        elif self.kind == 'item':
            # Cosine similarity between item rating vectors (columns of centered user ratings)
            # This is adjusted cosine similarity
            sim = cosine_similarity(self.user_item_centered.T.values)
            self.similarity_matrix = pd.DataFrame(sim, index=self.user_item.columns, columns=self.user_item.columns)
            
        print(f"{self.kind.capitalize()}-Based CF similarity matrix computed successfully.")

    def predict_rating(self, user_id, movie_id, user_ratings_dict=None):
        """
        Predicts a rating for user_id and movie_id.
        user_ratings_dict: Optional dict of {movieId: rating} for online/new user prediction.
        """
        # If movie is completely unknown, return global mean
        if self.user_item is not None and movie_id not in self.user_item.columns:
            return self.global_mean
            
        if self.kind == 'user':
            return self._predict_user_based(user_id, movie_id, user_ratings_dict)
        else:
            return self._predict_item_based(user_id, movie_id, user_ratings_dict)
            
    def _predict_user_based(self, user_id, movie_id, user_ratings_dict=None):
        # Determine target user details
        if user_ratings_dict is not None:
            # We have a custom user representation
            ratings_series = pd.Series(user_ratings_dict)
            u_mean = ratings_series.mean()
            u_centered = ratings_series - u_mean
            
            # Compute similarity of this custom user to all existing users
            u_vector = pd.Series(0, index=self.user_item.columns)
            u_vector.update(u_centered)
            
            # cosine similarity
            # u_vector shape: (1, num_movies), user_item_centered shape: (num_users, num_movies)
            existing_matrix = self.similarity_matrix.index
            sims = cosine_similarity(u_vector.values.reshape(1, -1), self.user_item_centered.values)[0]
            user_similarities = pd.Series(sims, index=self.user_item_centered.index)
        else:
            # Existing user
            if user_id not in self.user_item.index:
                return self.global_mean  # New user with no data
            u_mean = self.user_means.loc[user_id]
            user_similarities = self.similarity_matrix.loc[user_id]
            
        # Get users who rated this movie
        movie_ratings = self.user_item[movie_id].dropna()
        if movie_ratings.empty:
            return u_mean
            
        # Find intersection
        common_users = movie_ratings.index
        sim_scores = user_similarities.loc[common_users]
        
        # If the target user is in the list, exclude themselves
        if user_ratings_dict is None and user_id in sim_scores:
            sim_scores = sim_scores.drop(user_id)
            
        # Drop zero or negative similarities
        sim_scores = sim_scores[sim_scores > 0]
        if sim_scores.empty:
            return u_mean
            
        # Select top k neighbors
        top_neighbors = sim_scores.nlargest(self.k_neighbors)
        
        # Get ratings from top neighbors and center them
        neighbor_ratings = movie_ratings.loc[top_neighbors.index]
        neighbor_means = self.user_means.loc[top_neighbors.index]
        neighbor_centered_ratings = neighbor_ratings - neighbor_means
        
        # Weighted sum of centered ratings
        weighted_sum = np.dot(top_neighbors.values, neighbor_centered_ratings.values)
        sum_abs_sim = np.sum(np.abs(top_neighbors.values))
        
        if sum_abs_sim == 0:
            return u_mean
            
        predicted = u_mean + (weighted_sum / sum_abs_sim)
        return np.clip(predicted, 0.5, 5.0)

    def _predict_item_based(self, user_id, movie_id, user_ratings_dict=None):
        # Determine target user ratings
        if user_ratings_dict is not None:
            user_ratings = pd.Series(user_ratings_dict)
        else:
            if user_id not in self.user_item.index:
                return self.global_mean
            user_ratings = self.user_item.loc[user_id].dropna()
            
        if user_ratings.empty:
            return self.global_mean
            
        # Get similarities of target movie to all movies rated by this user
        rated_movie_ids = user_ratings.index
        if movie_id not in self.similarity_matrix.index:
            return self.global_mean
            
        sim_scores = self.similarity_matrix.loc[movie_id, rated_movie_ids]
        
        # If movie_id is in rated, we might exclude it (but normally we predict for unrated)
        if movie_id in sim_scores:
            sim_scores = sim_scores.drop(movie_id)
            
        # Filter positive similarities
        sim_scores = sim_scores[sim_scores > 0]
        if sim_scores.empty:
            # If no positive similarity, return user's mean rating
            return user_ratings.mean()
            
        # Take top k similar items
        top_items = sim_scores.nlargest(self.k_neighbors)
        
        # Weighted sum of user ratings
        target_ratings = user_ratings.loc[top_items.index]
        weighted_sum = np.dot(top_items.values, target_ratings.values)
        sum_sim = np.sum(top_items.values)
        
        if sum_sim == 0:
            return user_ratings.mean()
            
        predicted = weighted_sum / sum_sim
        return np.clip(predicted, 0.5, 5.0)

    def recommend(self, user_id, top_n=5, user_ratings_dict=None, exclude_rated=True):
        """
        Generates recommendations for a user.
        user_ratings_dict: Optional dict of {movieId: rating} for custom/new user recommendations.
        """
        # Get list of rated movie IDs to exclude
        if user_ratings_dict is not None:
            rated_movies = set(user_ratings_dict.keys())
        else:
            if user_id not in self.user_item.index:
                return []  # New user without history
            rated_movies = set(self.user_item.loc[user_id].dropna().index)
            
        predictions = []
        all_movies = self.user_item.columns
        
        for m_id in all_movies:
            if exclude_rated and m_id in rated_movies:
                continue
            pred_rating = self.predict_rating(user_id, m_id, user_ratings_dict)
            predictions.append((m_id, pred_rating))
            
        # Sort by predicted rating descending
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_n]

    def save(self, file_path):
        """Saves the collaborative filtering model."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump({
                'kind': self.kind,
                'k_neighbors': self.k_neighbors,
                'user_item': self.user_item,
                'user_means': self.user_means,
                'user_item_centered': self.user_item_centered,
                'similarity_matrix': self.similarity_matrix,
                'global_mean': self.global_mean
            }, f)
        print(f"Collaborative Filtering ({self.kind}) model saved to {file_path}")

    def load(self, file_path):
        """Loads the collaborative filtering model."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file {file_path} not found.")
            
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.kind = data['kind']
            self.k_neighbors = data['k_neighbors']
            self.user_item = data['user_item']
            self.user_means = data['user_means']
            self.user_item_centered = data['user_item_centered']
            self.similarity_matrix = data['similarity_matrix']
            self.global_mean = data['global_mean']
        print(f"Collaborative Filtering ({self.kind}) model loaded from {file_path}")
