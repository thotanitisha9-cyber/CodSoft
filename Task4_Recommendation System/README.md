# AI Recommendation System

An advanced, interactive Python-based movie recommendation system that utilizes multiple recommendation techniques: **Content-Based Filtering**, **Collaborative Filtering** (both User-Based and Item-Based), and a **Hybrid Recommender** engine. It features a premium, responsive **Streamlit** web dashboard for real-time recommendations, interactive ratings, and model performance visualization.

---

## 🚀 Key Features

* **Content-Based Filtering**: Recommends movies by building TF-IDF content profiles from genres, titles, and user tags, calculating similarity using Cosine Similarity.
* **Collaborative Filtering**:
  * **User-Based Collaborative Filtering**: Identifies users with similar tastes using Pearson correlation (mean-centered cosine similarity) and predicts ratings.
  * **Item-Based Collaborative Filtering**: Computes adjusted cosine similarities between movies based on user rating patterns to make predictions.
* **Hybrid Engine**: Custom blending of Content-Based and Collaborative Filtering predictions ($Score_{hybrid} = \alpha \cdot Score_{collaborative} + (1-\alpha) \cdot Score_{content}$).
* **Cold Start Handling**: Successfully handles new users with limited rating history by recommending popular movies using a **Bayesian Average** formula combined with user-selected favorite genres.
* **Interactive Streamlit UI**:
  * Select or create temporary profiles.
  * Search catalog and submit ratings in real-time.
  * View dynamic updates to recommendations.
  * Examine offline validation metrics (RMSE, MAE, Precision@K, Recall@K) on interactive charts.

---

## 📐 Recommendation Core Concepts

### 1. Content-Based Filtering
This model recommends items similar to those a user previously liked.
* **Feature Extraction**: Compiles a text representation ("metadata soup") for each movie: `Title` + `Genres` + `User Tags`.
* **Vectorization**: Transforms the soup using `TfidfVectorizer` to extract term frequency-inverse document frequency weights.
* **Similarity**: Calculates a square cosine similarity matrix:
  $$\text{Similarity}(A, B) = \frac{A \cdot B}{\|A\|\|B\|}$$
* **Scoring**: Predicts ratings or similarity weights, offsetting ratings by $2.5$ (neutral value) so that movies rated poorly by the user actively reduce scores for similar movies.

### 2. Collaborative Filtering
This technique recommends items based on user-item interaction histories.
* **User-Based Collaborative Filtering**:
  * Ratings are mean-centered per user to remove individual scale bias (e.g., generous vs. strict raters).
  * Missing ratings are predicted using:
    $$\hat{r}_{u, i} = \bar{r}_u + \frac{\sum_{v \in N_i(u)} \text{sim}(u, v) \cdot (r_{v, i} - \bar{r}_v)}{\sum_{v \in N_i(u)} |\text{sim}(u, v)|}$$
* **Item-Based Collaborative Filtering**:
  * Computes similarities between items (adjusted cosine similarity) using centered user ratings.
  * Predicts ratings via a weighted average of the user's ratings on similar items:
    $$\hat{r}_{u, i} = \frac{\sum_{j \in N_u(i)} \text{sim}(i, j) \cdot r_{u, j}}{\sum_{j \in N_u(i)} |\text{sim}(i, j)|}$$

### 3. Hybrid Recommender
Combines the strengths of content profiles and interaction patterns.
* Evaluates both CB and CF predictions and computes a weighted average.
* Overcomes the **sparsity** limitation of Collaborative Filtering and the **over-specialization** limitation of Content-Based models.

### 4. Cold Start via Bayesian Average
For new users with less than 2 ratings, standard collaborative filters cannot calculate tastes. The system falls back on a Bayesian Average popularity score:
$$\text{Bayesian Average} = \frac{v \cdot R + m \cdot C}{v + m}$$
Where:
* $v$ is the number of ratings for the movie.
* $R$ is the average rating of the movie.
* $m$ is the prior weight (minimum ratings required, default = 5).
* $C$ is the average rating across the entire dataset.
This prevents movies with only one 5-star rating from outranking classic blockbusters with thousands of 4.5-star ratings.

---

## 📂 Project Structure

```
RecommendationSystem/
│
├── dataset/                    # Stores datasets (movies, ratings, tags)
├── models/                     # Holds serialized pickle files & offline evaluation metrics
├── app.py                      # Streamlit web dashboard application
├── train.py                    # Model training, split validation, and evaluation pipeline
├── recommender.py              # Hybrid recommender wrapper & cold start logic
├── content_based.py            # Content-Based Recommender (TF-IDF + Cosine Similarity)
├── collaborative_filtering.py  # Collaborative Filtering Recommender (User/Item-Based CF)
├── utils.py                    # Dataset downloaders, synthetic fallback generator & metrics
├── requirements.txt            # Project dependencies list
└── README.md                   # Project documentation (this file)
```

---

## 🛠️ Installation Steps

1. **Ensure Python is installed** (Python 3.8+ recommended).
2. **Clone/extract the project directory** and open a terminal inside the project root folder:
   ```bash
   cd CodSoft-RS
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ How to Run

### Step 1: Run the Training Pipeline
Run the training script to fetch the MovieLens dataset (or create a synthetic fallback dataset if offline), train all recommendation models, compute evaluation metrics, and save serialized files:
```bash
python train.py
```
*Output files will be saved in `models/` (including `metadata.pkl`, model pickles, and `metrics.json`).*

### Step 2: Run the Streamlit Dashboard
Launch the web interface locally:
```bash
streamlit run app.py
```
Open the local URL displayed in your browser (typically `http://localhost:8501`).

---

## 📊 Evaluation Metrics
The training script splits interactions into 80% train and 20% test sets. It computes and saves the following metrics:
* **RMSE** (Root Mean Squared Error): Measures rating prediction accuracy.
* **MAE** (Mean Absolute Error): Measures average rating prediction magnitude error.
* **Precision@K**: Measures recommendation relevance (fraction of top $K$ suggestions liked by the user).
* **Recall@K**: Measures catalog coverage (fraction of user's liked movies successfully recommended).

---

## 🔮 Future Improvements

1. **Matrix Factorization (SVD)**: Integrate SVD or Alternating Least Squares (ALS) using sparse matrix libraries to scale collaborative filtering.
2. **Deep Learning Recommenders**: Integrate neural network architectures like Neural Collaborative Filtering (NCF) or autoencoders.
3. **Real-time Session-Based recommendations**: Use Gated Recurrent Units (GRU) or Transformers to suggest items based on session click trails.
4. **Enriched Movie Metadata**: Fetch movie poster images and details dynamically using the TMDB API.
