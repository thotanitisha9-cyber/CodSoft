import os
import json
import pandas as pd
import numpy as np
from utils import download_and_prepare_dataset, split_ratings_train_test, rmse, mae, precision_recall_at_k
from recommender import HybridRecommender

MODELS_DIR = "models"
METRICS_FILE = os.path.join(MODELS_DIR, "metrics.json")

def evaluate_models(recommender_train, train_df, test_df, k=5, like_threshold=3.5):
    """
    Evaluates the trained models on the test set.
    """
    print("\nEvaluating models on test set...")
    
    # Pre-build user ratings dictionary from training set for content-based predictions
    user_train_ratings = {}
    for u_id, group in train_df.groupby('userId'):
        user_train_ratings[u_id] = dict(zip(group['movieId'], group['rating']))
        
    # Gather actual likes in test set for Precision@K and Recall@K
    actual_likes = {}
    for u_id, group in test_df.groupby('userId'):
        liked_movies = set(group[group['rating'] >= like_threshold]['movieId'].values)
        if liked_movies:
            actual_likes[u_id] = liked_movies
            
    # List of test rating tuples: (userId, movieId, true_rating)
    test_ratings = list(zip(test_df['userId'], test_df['movieId'], test_df['rating']))
    
    results = {}
    model_configs = [
        ('Content-Based', lambda u, m: recommender_train.content_model.predict_rating(user_train_ratings.get(u, {}), m)),
        ('CF User-Based', lambda u, m: recommender_train.cf_user_model.predict_rating(u, m)),
        ('CF Item-Based', lambda u, m: recommender_train.cf_item_model.predict_rating(u, m)),
        ('Hybrid (Item-CF + Content)', lambda u, m: recommender_train.predict_rating(u, m, user_train_ratings.get(u, {}), cf_kind='item', alpha=0.5))
    ]
    
    # 1. Rating prediction evaluation (RMSE, MAE)
    for model_name, predict_func in model_configs:
        print(f"Evaluating rating predictions for {model_name}...")
        y_true = []
        y_pred = []
        
        for u_id, m_id, rating in test_ratings:
            pred = predict_func(u_id, m_id)
            y_true.append(rating)
            y_pred.append(pred)
            
        results[model_name] = {
            'RMSE': float(rmse(y_true, y_pred)),
            'MAE': float(mae(y_true, y_pred))
        }
        
    # 2. Recommendation quality evaluation (Precision@K & Recall@K)
    # To speed up calculation, we evaluate a subset of users or all users if the dataset is small
    print(f"Evaluating top-{k} recommendations (Precision@{k} & Recall@{k})...")
    test_users = list(actual_likes.keys())
    
    # Initialize recommendation dicts
    recs_cb = {}
    recs_cf_user = {}
    recs_cf_item = {}
    recs_hybrid = {}
    
    count = 0
    total_users = len(test_users)
    
    for u_id in test_users:
        count += 1
        if count % 20 == 0:
            print(f"  Processed {count}/{total_users} users...")
            
        # Get user's ratings in training set
        u_ratings = user_train_ratings.get(u_id, {})
        
        # Recommendations for Content-Based
        recs_cb[u_id] = [m for m, _ in recommender_train.content_model.recommend(u_ratings, top_n=k, exclude_rated=True)]
        
        # Recommendations for Collaborative Filtering (User-Based)
        recs_cf_user[u_id] = [m for m, _ in recommender_train.cf_user_model.recommend(u_id, top_n=k, exclude_rated=True)]
        
        # Recommendations for Collaborative Filtering (Item-Based)
        recs_cf_item[u_id] = [m for m, _ in recommender_train.cf_item_model.recommend(u_id, top_n=k, exclude_rated=True)]
        
        # Recommendations for Hybrid
        recs_hybrid[u_id] = [m for m, _ in recommender_train.recommend(u_id, top_n=k, user_ratings_dict=u_ratings, cf_kind='item', alpha=0.5, exclude_rated=True)]
        
    # Compute Precision & Recall
    prec_cb, rec_cb = precision_recall_at_k(recs_cb, actual_likes, k)
    prec_cf_user, rec_cf_user = precision_recall_at_k(recs_cf_user, actual_likes, k)
    prec_cf_item, rec_cf_item = precision_recall_at_k(recs_cf_item, actual_likes, k)
    prec_hybrid, rec_hybrid = precision_recall_at_k(recs_hybrid, actual_likes, k)
    
    results['Content-Based'][f'Precision@{k}'] = float(prec_cb)
    results['Content-Based'][f'Recall@{k}'] = float(rec_cb)
    
    results['CF User-Based'][f'Precision@{k}'] = float(prec_cf_user)
    results['CF User-Based'][f'Recall@{k}'] = float(rec_cf_user)
    
    results['CF Item-Based'][f'Precision@{k}'] = float(prec_cf_item)
    results['CF Item-Based'][f'Recall@{k}'] = float(rec_cf_item)
    
    results['Hybrid (Item-CF + Content)'][f'Precision@{k}'] = float(prec_hybrid)
    results['Hybrid (Item-CF + Content)'][f'Recall@{k}'] = float(rec_hybrid)
    
    # Print results summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS SUMMARY")
    print("="*50)
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        for metric_name, val in metrics.items():
            print(f"  {metric_name}: {val:.4f}")
    print("="*50)
    
    return results

def main():
    print("Starting Machine Learning Pipeline...")
    
    # 1. Load dataset (Download from MovieLens or generate synthetic fallback)
    movies_path, ratings_path, tags_path = download_and_prepare_dataset()
    
    # Read files
    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    
    tags_df = None
    if os.path.exists(tags_path):
        tags_df = pd.read_csv(tags_path)
        
    print(f"Loaded {len(movies_df)} movies and {len(ratings_df)} rating records.")
    
    # Print dataset statistics
    num_users = ratings_df['userId'].nunique()
    num_movies = ratings_df['movieId'].nunique()
    sparsity = 100 * (1 - len(ratings_df) / (num_users * num_movies))
    print(f"Unique Users: {num_users}")
    print(f"Unique Movies in Ratings: {num_movies}")
    print(f"Interaction Matrix Sparsity: {sparsity:.2f}%")
    
    # 2. Split into Train & Test sets (80% Train, 20% Test)
    train_df, test_df = split_ratings_train_test(ratings_df, test_ratio=0.2, random_state=42)
    print(f"Split data into train size: {len(train_df)} and test size: {len(test_df)}")
    
    # 3. Train models on Training Set and Evaluate
    recommender_eval = HybridRecommender()
    # Fit on training set
    recommender_eval.fit(movies_df, train_df, tags_df)
    
    # Evaluate
    results = evaluate_models(recommender_eval, train_df, test_df, k=5, like_threshold=3.5)
    
    # Save evaluation results to json file
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(METRICS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation metrics saved to {METRICS_FILE}")
    
    # 4. Train final models on FULL dataset and save to disk
    print("\nTraining final recommendation models on complete dataset...")
    recommender_final = HybridRecommender()
    recommender_final.fit(movies_df, ratings_df, tags_df)
    
    recommender_final.save(MODELS_DIR)
    print("Pipeline completed successfully! All models saved to models/ directory.")

if __name__ == "__main__":
    main()
