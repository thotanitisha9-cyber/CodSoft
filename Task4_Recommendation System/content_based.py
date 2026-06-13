import os
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.movies_df = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        
    def _prepare_metadata(self, movies_df, tags_df=None):
        """
        Cleans genres, aggregates tags, and creates a combined text representation ('soup')
        for each movie.
        """
        df = movies_df.copy()
        
        # Clean genres (replace '|' with spaces)
        df['cleaned_genres'] = df['genres'].str.replace('|', ' ', regex=False)
        df['cleaned_genres'] = df['cleaned_genres'].fillna('')
        
        # Aggregate tags per movie
        if tags_df is not None and not tags_df.empty:
            # Group tags by movieId and join as string
            tags_agg = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
            tags_agg.rename(columns={'tag': 'aggregated_tags'}, inplace=True)
            df = df.merge(tags_agg, on='movieId', how='left')
            df['aggregated_tags'] = df['aggregated_tags'].fillna('')
        else:
            df['aggregated_tags'] = ''
            
        # Extract title words (strip year if needed, but keeping it is fine)
        df['cleaned_title'] = df['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
        
        # Create metadata soup: combined title, genres, and tags
        df['soup'] = df['cleaned_title'] + " " + df['cleaned_genres'] + " " + df['aggregated_tags']
        df['soup'] = df['soup'].str.lower().fillna('')
        
        return df

    def fit(self, movies_df, tags_df=None):
        """
        Fits TF-IDF on the movie metadata soup and computes movie similarities.
        """
        print("Fitting Content-Based Recommender...")
        self.movies_df = self._prepare_metadata(movies_df, tags_df)
        
        # Re-build index mappings
        self.movie_id_to_idx = {row['movieId']: idx for idx, row in self.movies_df.iterrows()}
        self.idx_to_movie_id = {idx: row['movieId'] for idx, row in self.movies_df.iterrows()}
        
        # Compute TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies_df['soup'])
        
        # Compute cosine similarity
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        print("Content-Based similarity matrix built successfully.")
        
    def predict_rating(self, user_ratings, movie_id):
        """
        Predicts a rating for a specific movie based on user's rating history.
        user_ratings: dict of {movieId: rating}
        movie_id: ID of the movie to predict rating for
        """
        if movie_id not in self.movie_id_to_idx:
            return 2.5  # Neutral fallback rating if movie is unknown
            
        target_idx = self.movie_id_to_idx[movie_id]
        similarities = self.similarity_matrix[target_idx]
        
        weighted_sum = 0.0
        similarity_sum = 0.0
        
        for rated_movie_id, rating in user_ratings.items():
            if rated_movie_id == movie_id:
                return rating  # If they already rated it, return it
                
            if rated_movie_id in self.movie_id_to_idx:
                rated_idx = self.movie_id_to_idx[rated_movie_id]
                sim = similarities[rated_idx]
                
                # We only consider positive similarity
                if sim > 0:
                    weighted_sum += sim * rating
                    similarity_sum += sim
                    
        if similarity_sum == 0.0:
            return 2.5  # Fallback if no similar rated movies
            
        return weighted_sum / similarity_sum

    def recommend(self, user_ratings, top_n=5, exclude_rated=True):
        """
        Generates recommendations for a user based on their current ratings.
        user_ratings: dict of {movieId: rating}
        """
        if not user_ratings:
            # Cold start: Return popular movies (handled by wrapper, but fallback here)
            return []
            
        scores = []
        rated_movies = set(user_ratings.keys())
        
        for movie_id in self.movies_df['movieId'].values:
            if exclude_rated and movie_id in rated_movies:
                continue
                
            # Score calculation: sum of similarity * (rating - 2.5)
            # Offset rating by 2.5 so that negative ratings (e.g. 1.0) suppress similar movies
            target_idx = self.movie_id_to_idx[movie_id]
            similarities = self.similarity_matrix[target_idx]
            
            score = 0.0
            total_sim = 0.0
            
            for rated_movie_id, rating in user_ratings.items():
                if rated_movie_id in self.movie_id_to_idx:
                    rated_idx = self.movie_id_to_idx[rated_movie_id]
                    sim = similarities[rated_idx]
                    
                    if sim > 0:
                        # Offset rating by 2.5 (neutral point)
                        score += sim * (rating - 2.5)
                        total_sim += sim
                        
            # Normalize the score to keep it relative
            final_score = score / (total_sim + 1e-8)
            scores.append((movie_id, final_score))
            
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    def save(self, file_path):
        """Saves the fitted model to disk."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'movies_df': self.movies_df,
                'tfidf_matrix': self.tfidf_matrix,
                'similarity_matrix': self.similarity_matrix,
                'movie_id_to_idx': self.movie_id_to_idx,
                'idx_to_movie_id': self.idx_to_movie_id
            }, f)
        print(f"Content-Based model saved to {file_path}")

    def load(self, file_path):
        """Loads a saved model from disk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file {file_path} not found.")
            
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.movies_df = data['movies_df']
            self.tfidf_matrix = data['tfidf_matrix']
            self.similarity_matrix = data['similarity_matrix']
            self.movie_id_to_idx = data['movie_id_to_idx']
            self.idx_to_movie_id = data['idx_to_movie_id']
        print(f"Content-Based model loaded from {file_path}")
