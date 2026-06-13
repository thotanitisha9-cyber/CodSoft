import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import altair as altair
from recommender import HybridRecommender
from utils import download_and_prepare_dataset

# Set page configuration with a premium dark-themed layout
st.set_page_config(
    page_title="AI Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom styling for a premium modern appearance
st.markdown("""
<style>
    /* Main container adjustments */
    .reportview-container {
        background: #0e1117;
    }
    
    /* Premium Title Styling */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(90deg, #ff7e5f, #feb47b, #86a8e7, #7f7fd5);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 10s ease infinite;
        font-weight: 800;
        font-size: 3.2rem;
        margin-bottom: 5px;
    }
    
    /* Subtitle style */
    .subtitle {
        color: #8892b0;
        font-size: 1.15rem;
        margin-bottom: 25px;
        font-weight: 300;
    }
    
    /* Movie Card Container */
    .movie-card {
        background: rgba(30, 34, 48, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 25px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        min-height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        border-color: rgba(134, 168, 231, 0.5);
        box-shadow: 0 8px 30px 0 rgba(134, 168, 231, 0.25);
    }
    
    /* Card details */
    .movie-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #eceff4;
        margin-bottom: 6px;
        line-height: 1.3;
    }
    .movie-genres {
        font-size: 0.85rem;
        color: #88c0d0;
        font-weight: 500;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .movie-score {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 15px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Emojis matching genre */
    .rating-badge {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: #ffffff;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 30px;
    }
    
    .match-percent {
        color: #a3be8c;
        font-weight: 700;
        font-size: 0.9rem;
    }
    
    /* Keyframe animations */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
</style>
""", unsafe_allow_html=True)

# Helper function to assign visual emojis to movie genres
def get_genre_emojis(genre_str):
    mapping = {
        "action": "💥",
        "adventure": "🧭",
        "animation": "🦄",
        "children": "👶",
        "comedy": "😂",
        "crime": "🕵️",
        "drama": "🎭",
        "fantasy": "🔮",
        "horror": "👻",
        "musical": "🎵",
        "mystery": "🔎",
        "romance": "❤️",
        "sci-fi": "🚀",
        "thriller": "😱",
        "war": "⚔️",
        "western": "🤠",
        "documentary": "📹"
    }
    genres = genre_str.split('|')
    emojis = []
    for g in genres:
        key = g.lower().strip()
        if key in mapping:
            emojis.append(f"{mapping[key]} {g}")
        else:
            emojis.append(f"🎬 {g}")
    return " | ".join(emojis)

MODELS_DIR = "models"
METRICS_FILE = os.path.join(MODELS_DIR, "metrics.json")

# 1. Initialize System State
@st.cache_resource
def load_recommender():
    recommender = HybridRecommender()
    try:
        recommender.load(MODELS_DIR)
        return recommender, True
    except Exception as e:
        st.warning(f"Could not load pre-trained models from {MODELS_DIR}. Error: {e}")
        return recommender, False

recommender, is_loaded = load_recommender()

# Check if datasets exist; otherwise, download or generate them
if not os.path.exists("dataset/movies.csv") or not os.path.exists("dataset/ratings.csv"):
    download_and_prepare_dataset()

# Initialize ratings database and movie catalogs in Session State
if 'movies_df' not in st.session_state:
    st.session_state.movies_df = pd.read_csv("dataset/movies.csv")

# Extract all unique genres for filtering and multiselects (global scope)
all_genres = set()
for g_list in st.session_state.movies_df['genres'].str.split('|'):
    if isinstance(g_list, list):
        all_genres.update(g_list)
all_genres = sorted(list(all_genres))
    
if 'ratings_df' not in st.session_state:
    # If model is loaded, pull current ratings from it; otherwise read csv
    if is_loaded and recommender.ratings_df is not None:
        st.session_state.ratings_df = recommender.ratings_df.copy()
    else:
        st.session_state.ratings_df = pd.read_csv("dataset/ratings.csv")

if 'tags_df' not in st.session_state:
    if os.path.exists("dataset/tags.csv"):
        st.session_state.tags_df = pd.read_csv("dataset/tags.csv")
    else:
        st.session_state.tags_df = pd.DataFrame(columns=['userId', 'movieId', 'tag', 'timestamp'])

# If we have a local ratings update, we keep a custom user ratings dictionary for active session users
if 'session_user_ratings' not in st.session_state:
    st.session_state.session_user_ratings = {}

# Header Section
st.markdown("<h1 class='main-title'>Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A complete modular AI recommendation engine blending Content-Based Filtering & Collaborative Filtering.</div>", unsafe_allow_html=True)

# 2. Side Panel Configuration
st.sidebar.image("https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=400", use_column_width=True)
st.sidebar.markdown("### 🛠️ Configuration Console")

# User setup choice
user_mode = st.sidebar.radio(
    "Select User Mode",
    ["Existing User Profile", "Create New User (Cold Start)"]
)

# Extract user lists
existing_users = sorted(st.session_state.ratings_df['userId'].unique().tolist())

if user_mode == "Existing User Profile":
    selected_user_id = st.sidebar.selectbox(
        "Select User ID",
        existing_users,
        help="Choose a user profile to load historical ratings and generate recommendations."
    )
    
    # Initialize ratings from dataframe if not in session cache
    if selected_user_id not in st.session_state.session_user_ratings:
        user_history_df = st.session_state.ratings_df[st.session_state.ratings_df['userId'] == selected_user_id]
        st.session_state.session_user_ratings[selected_user_id] = dict(zip(user_history_df['movieId'], user_history_df['rating']))
        
    user_ratings = st.session_state.session_user_ratings[selected_user_id]
    
else:
    selected_user_id = -999  # Code for custom cold-start user
    st.sidebar.info("💡 Create a temporary profile to test recommendations with limited or custom ratings.")
    
    # Let user pick favorite genres
    fav_genres = st.sidebar.multiselect(
        "Select Favorite Genres",
        all_genres,
        default=all_genres[:2] if len(all_genres) >= 2 else []
    )
    
    # Custom user ratings container
    if 'cold_start_ratings' not in st.session_state:
        st.session_state.cold_start_ratings = {}
    user_ratings = st.session_state.cold_start_ratings

# Select Recommendation Technique
rec_technique = st.sidebar.selectbox(
    "Recommendation Method",
    [
        "Hybrid Recommender",
        "Content-Based Filtering",
        "User-Based Collaborative Filtering",
        "Item-Based Collaborative Filtering"
    ]
)

# Detail toggles depending on method
alpha = 0.5
cf_kind = 'item'

if rec_technique == "Hybrid Recommender":
    cf_kind = st.sidebar.selectbox("Collaborative Engine for Hybrid", ["item", "user"])
    alpha = st.sidebar.slider(
        "Collaborative Weight (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Higher values favor Collaborative predictions; lower values favor Content-Based similarities."
    )
    st.sidebar.caption("Formula: α * Collaborative + (1-α) * Content")

elif rec_technique == "User-Based Collaborative Filtering":
    cf_kind = 'user'
elif rec_technique == "Item-Based Collaborative Filtering":
    cf_kind = 'item'

top_k_slider = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

# Build/Retrain button in sidebar if not trained
if not is_loaded:
    st.sidebar.error("⚠️ Model files are not trained!")
    if st.sidebar.button("⚙️ Run Model Trainer Now"):
        with st.spinner("Executing train.py... This may take a minute."):
            import subprocess
            res = subprocess.run(["python", "train.py"], capture_output=True, text=True)
            st.sidebar.success("Training complete! Reloading page...")
            st.rerun()

# 3. Main Dashboard Tabs
tab_recs, tab_profile, tab_explore, tab_eval = st.tabs([
    "🎯 Recommendations",
    "👤 User Profile & Rating History",
    "🔍 Explore & Search Movie Catalog",
    "📈 Model Evaluation Metrics"
])

# ================= TAB: RECOMMENDATIONS =================
with tab_recs:
    st.subheader("📋 Top Recommendations for You")
    
    # If cold start, display preferences
    if user_mode == "Create New User (Cold Start)":
        st.markdown(f"**Selected Genres**: {', '.join(fav_genres) if fav_genres else 'None'}")
        st.markdown(f"**Ratings Submitted**: {len(user_ratings)} movies rated.")
        
        # If user has no ratings and no genres selected, show a prompt
        if not fav_genres and not user_ratings:
            st.warning("Please select favorite genres in the sidebar or rate some movies in the tabs to get personalized suggestions.")
            
    # Generate recommendations logic
    if st.button("⚡ Generate Recommendations", type="primary"):
        # Fit models dynamically if not fit
        if not is_loaded:
            # Fit on currently loaded data
            recommender.fit(st.session_state.movies_df, st.session_state.ratings_df, st.session_state.tags_df)
            
        with st.spinner("Running recommendation algorithms..."):
            recommendations = []
            
            # Check for Cold Start (New User with less than 2 ratings and has selected genres)
            if user_mode == "Create New User (Cold Start)" and len(user_ratings) < 2:
                # Use cold-start genre/popularity recommendation
                recs_df = recommender.get_cold_start_recommendations(
                    favorite_genres=fav_genres,
                    top_n=top_k_slider
                )
                recommendations = [(row['movieId'], row['bayesian_rating']) for _, row in recs_df.iterrows()]
                st.info("ℹ️ Showing genre-filtered popular movies (Bayesian rating rank) because you have less than 2 ratings (Cold Start).")
                
            else:
                # Standard recommendation methods
                u_id = selected_user_id
                
                # Fetch recommendations based on method
                if rec_technique == "Content-Based Filtering":
                    # For content-based, if new user, pass custom dict
                    recommendations = recommender.content_model.recommend(
                        user_ratings=user_ratings,
                        top_n=top_k_slider
                    )
                elif rec_technique == "User-Based Collaborative Filtering":
                    recommendations = recommender.cf_user_model.recommend(
                        user_id=u_id,
                        top_n=top_k_slider,
                        user_ratings_dict=user_ratings if user_mode == "Create New User (Cold Start)" else None
                    )
                elif rec_technique == "Item-Based Collaborative Filtering":
                    recommendations = recommender.cf_item_model.recommend(
                        user_id=u_id,
                        top_n=top_k_slider,
                        user_ratings_dict=user_ratings if user_mode == "Create New User (Cold Start)" else None
                    )
                else:  # Hybrid
                    recommendations = recommender.recommend(
                        user_id=u_id,
                        top_n=top_k_slider,
                        user_ratings_dict=user_ratings if user_mode == "Create New User (Cold Start)" else None,
                        cf_kind=cf_kind,
                        alpha=alpha
                    )
                    
            if not recommendations:
                # Fallback to popular movies
                popular = recommender.get_popular_movies(top_n=top_k_slider)
                recommendations = [(row['movieId'], row['bayesian_rating']) for _, row in popular.iterrows()]
                st.warning("No recommendations could be computed. Displaying popular movies as fallback.")
                
            # Render recommended movies in columns
            cols = st.columns(2)
            for i, (m_id, score) in enumerate(recommendations):
                # Retrieve movie info
                movie_info = st.session_state.movies_df[st.session_state.movies_df['movieId'] == m_id]
                if movie_info.empty:
                    continue
                row = movie_info.iloc[0]
                
                # Format scores/ratings
                if rec_technique in ["User-Based Collaborative Filtering", "Item-Based Collaborative Filtering", "Hybrid Recommender"] and user_mode == "Existing User Profile":
                    score_label = f"⭐ Predicted Rating: {score:.2f} / 5.0"
                    match_percent = int(np.clip((score / 5.0) * 100, 10, 99))
                else:
                    score_label = f"🔥 Score Rank: {score:.2f}"
                    match_percent = int(np.clip(100 - (i * 3), 50, 98))
                    
                genre_styled = get_genre_emojis(row['genres'])
                
                # Render in column (alternating)
                col = cols[i % 2]
                
                card_html = f"""
                <div class="movie-card">
                    <div>
                        <div class="movie-title">{i+1}. {row['title']}</div>
                        <div class="movie-genres">{genre_styled}</div>
                    </div>
                    <div class="movie-score">
                        <span class="rating-badge">{score_label}</span>
                        <span class="match-percent">{match_percent}% Match</span>
                    </div>
                </div>
                """
                col.markdown(card_html, unsafe_allow_html=True)
    else:
        st.info("👈 Click the **Generate Recommendations** button on the sidebar to display your personalized results.")

# ================= TAB: USER PROFILE & HISTORY =================
with tab_profile:
    st.subheader("👤 User Rating Profile")
    
    if user_mode == "Existing User Profile":
        st.markdown(f"#### Active Profile: **User ID {selected_user_id}**")
        st.write(f"This user has rated **{len(user_ratings)}** movies in the dataset.")
        
        # Display existing ratings
        if user_ratings:
            history_list = []
            for m_id, r in user_ratings.items():
                m_title = st.session_state.movies_df[st.session_state.movies_df['movieId'] == m_id]['title'].values
                m_genres = st.session_state.movies_df[st.session_state.movies_df['movieId'] == m_id]['genres'].values
                title_str = m_title[0] if len(m_title) > 0 else f"Movie ID {m_id}"
                genres_str = m_genres[0] if len(m_genres) > 0 else ""
                history_list.append({"Movie ID": m_id, "Movie Title": title_str, "Genres": genres_str, "Rating": r})
                
            history_df = pd.DataFrame(history_list).sort_values(by="Rating", ascending=False)
            st.dataframe(history_df, use_container_width=True)
            
            # Simple ratings count plot
            chart_df = history_df['Rating'].value_counts().reset_index()
            chart_df.columns = ['Rating Value', 'Count']
            
            hist_chart = altair.Chart(chart_df).mark_bar(color='#feb47b', cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                x=altair.X('Rating Value:N', title='Rating Value'),
                y=altair.Y('Count:Q', title='Number of Ratings')
            ).properties(
                title="User's Rating Distribution",
                width=600,
                height=250
            )
            st.altair_chart(hist_chart, use_container_width=True)
        else:
            st.write("No ratings recorded for this user.")
            
    else:
        st.markdown("#### Active Profile: **Temporary Cold-Start User**")
        st.write("Submit movie ratings using the explore tab to populate this list and enable collaborative predictions.")
        if user_ratings:
            history_list = []
            for m_id, r in user_ratings.items():
                m_title = st.session_state.movies_df[st.session_state.movies_df['movieId'] == m_id]['title'].values
                m_genres = st.session_state.movies_df[st.session_state.movies_df['movieId'] == m_id]['genres'].values
                title_str = m_title[0] if len(m_title) > 0 else f"Movie ID {m_id}"
                genres_str = m_genres[0] if len(m_genres) > 0 else ""
                history_list.append({"Movie ID": m_id, "Movie Title": title_str, "Genres": genres_str, "Rating": r})
            st.dataframe(pd.DataFrame(history_list), use_container_width=True)
        else:
            st.warning("You haven't rated any movies yet. Go to the **Explore** tab, search for movies, and add your ratings!")

# ================= TAB: EXPLORE & SEARCH =================
with tab_explore:
    st.subheader("🔍 Explore Movie Catalog")
    st.write("Search the catalog to discover movies, inspect genre tags, and rate them to update predictions in real-time.")
    
    # Search controls
    search_query = st.text_input("🔍 Search movies by title", "")
    genre_filter = st.selectbox("Filter by Genre", ["All"] + all_genres)
    
    filtered_catalog = st.session_state.movies_df.copy()
    
    if search_query:
        filtered_catalog = filtered_catalog[filtered_catalog['title'].str.contains(search_query, case=False, na=False)]
        
    if genre_filter != "All":
        filtered_catalog = filtered_catalog[filtered_catalog['genres'].str.contains(genre_filter, case=False, na=False)]
        
    # Rate Movie form
    st.markdown("---")
    st.markdown("### 🌟 Submit a Rating")
    
    rate_col1, rate_col2, rate_col3 = st.columns([2, 1, 1])
    
    # Get movie options for dropdown based on filter
    movie_options = dict(zip(filtered_catalog['movieId'], filtered_catalog['title']))
    
    if not movie_options:
        rate_col1.warning("No movies match your search query/genre filter.")
    else:
        selected_rate_movie_id = rate_col1.selectbox(
            "Select Movie to Rate",
            options=list(movie_options.keys()),
            format_func=lambda x: movie_options[x]
        )
        
        # Current rating if any
        current_r = user_ratings.get(selected_rate_movie_id, 3.0)
        
        new_rating = rate_col2.slider(
            "Rating",
            min_value=0.5,
            max_value=5.0,
            value=float(current_r),
            step=0.5
        )
        
        if rate_col3.button("📝 Submit Rating", type="secondary", use_container_width=True):
            if user_mode == "Existing User Profile":
                # Update session cache
                st.session_state.session_user_ratings[selected_user_id][selected_rate_movie_id] = new_rating
                
                # Write back to st.session_state.ratings_df for collaborative model to use
                # Check if rating already exists in ratings_df
                mask = (st.session_state.ratings_df['userId'] == selected_user_id) & (st.session_state.ratings_df['movieId'] == selected_rate_movie_id)
                if mask.any():
                    st.session_state.ratings_df.loc[mask, 'rating'] = new_rating
                else:
                    new_row = pd.DataFrame([{
                        'userId': int(selected_user_id),
                        'movieId': int(selected_rate_movie_id),
                        'rating': float(new_rating),
                        'timestamp': 1618224000
                    }])
                    st.session_state.ratings_df = pd.concat([st.session_state.ratings_df, new_row], ignore_index=True)
                    
                # Dynamically fit the collaborative model internals to reflect this rating
                # This ensures real-time updates without retraining everything from scratch
                if is_loaded:
                    # Update training references
                    recommender.ratings_df = st.session_state.ratings_df.copy()
                    recommender.cf_user_model.fit(recommender.ratings_df)
                    recommender.cf_item_model.fit(recommender.ratings_df)
                    
                st.success(f"Successfully rated **{movie_options[selected_rate_movie_id]}** as **{new_rating}** stars!")
            else:
                # Cold-start custom user
                st.session_state.cold_start_ratings[selected_rate_movie_id] = new_rating
                st.success(f"Added rating of **{new_rating}** for **{movie_options[selected_rate_movie_id]}** to temporary profile!")
            st.rerun()

    # Display Movie Catalog Grid
    st.markdown("### 🎞️ Catalog Preview")
    st.dataframe(filtered_catalog[['movieId', 'title', 'genres']].head(100), use_container_width=True)

# ================= TAB: MODEL PERFORMANCE =================
with tab_eval:
    st.subheader("📈 Offline Validation Performance")
    st.write("These metrics represent model performance evaluated on a held-out test set (20% of ratings) using standard accuracy indicators.")
    
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, 'r') as f:
                metrics_data = json.load(f)
                
            # Convert JSON metrics to a flat dataframe for Altair plotting
            flat_metrics = []
            for model_name, model_metrics in metrics_data.items():
                for m_name, val in model_metrics.items():
                    flat_metrics.append({
                        'Model': model_name,
                        'Metric': m_name,
                        'Value': val
                    })
                    
            metrics_df = pd.DataFrame(flat_metrics)
            
            # Show Metrics table
            st.markdown("### 📊 Metrics Table")
            col_tbl, col_spacer = st.columns([2, 1])
            
            # Pivot table for clean display
            pivoted_df = metrics_df.pivot(index='Model', columns='Metric', values='Value').reset_index()
            col_tbl.dataframe(pivoted_df, use_container_width=True)
            
            # Split metrics into error-based (RMSE, MAE) and rank-based (Precision, Recall) for plotting
            st.markdown("### 📉 Metrics Visualizer")
            plot_col1, plot_col2 = st.columns(2)
            
            # Chart 1: Error Metrics (Lower is better)
            error_df = metrics_df[metrics_df['Metric'].isin(['RMSE', 'MAE'])]
            chart1 = altair.Chart(error_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=altair.X('Model:N', axis=altair.Axis(labelAngle=-15)),
                y=altair.Y('Value:Q', title='Error Value'),
                color=altair.Color('Metric:N', scale=altair.Scale(range=['#4C566A', '#ff7e5f'])),
                xOffset='Metric:N'
            ).properties(
                title="Rating Prediction Error (RMSE & MAE) - Lower is Better",
                height=350
            )
            plot_col1.altair_chart(chart1, use_container_width=True)
            
            # Chart 2: Rank Metrics (Higher is better)
            rank_df = metrics_df[~metrics_df['Metric'].isin(['RMSE', 'MAE'])]
            chart2 = altair.Chart(rank_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=altair.X('Model:N', axis=altair.Axis(labelAngle=-15)),
                y=altair.Y('Value:Q', title='Metric Value (0.0 to 1.0)'),
                color=altair.Color('Metric:N', scale=altair.Scale(range=['#86a8e7', '#a3be8c'])),
                xOffset='Metric:N'
            ).properties(
                title="Recommendation Quality (Precision@K & Recall@K)",
                height=350
            )
            plot_col2.altair_chart(chart2, use_container_width=True)
            
            # Metric explanations
            st.markdown("""
            #### 🧠 Key Concept Definitions
            * **RMSE (Root Mean Squared Error)**: Standard metric measuring the square root of the average squared differences between predicted ratings and actual ratings. Punishes large errors heavily.
            * **MAE (Mean Absolute Error)**: Measures the average magnitude of absolute errors in rating predictions. More robust to outliers.
            * **Precision@K**: The proportion of recommended items in the top-K suggestions that are relevant to the user (e.g. rated >= 3.5 in the test set).
            * **Recall@K**: The proportion of all relevant items in the test set that were successfully captured in the top-K recommendations.
            """)
        except Exception as e:
            st.error(f"Error parsing metrics.json: {e}")
    else:
        st.info("ℹ️ No offline metrics found. Execute the training pipeline (`train.py`) to generate evaluations.")
        if st.button("⚙️ Execute Model Pipeline Now", key="tab_retrain_btn"):
            with st.spinner("Executing train.py..."):
                import subprocess
                subprocess.run(["python", "train.py"])
                st.success("Evaluation pipeline finished. Refreshing...")
                st.rerun()
