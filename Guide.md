# Netflix Recommendation System with Causal AI + Uplift Modeling
## Complete Theory, Code Mapping, and Viva Voce Master Guide (Part 1 & Part 2)

---

### Table of Contents
- **PART 1 — BASELINE RECOMMENDATION SYSTEM**
  1. [Project Overview & Conceptual Framing](#1-project-overview--conceptual-framing)
  2. [Part 1 Code & Theory Mapping (Cells 1–14)](#2-part-1-section-by-section-code--theory-mapping)
     - [Cell 1: Dataset Ingestion](#cell-1-dataset-ingestion)
     - [Cell 2: Statistics & Sparsity Computation](#cell-2-statistics--sparsity-computation)
     - [Cell 3: User-Aware Train/Test Split](#cell-3-user-aware-traintest-split)
     - [Cell 4: User-Item Matrix Construction](#cell-4-user-item-matrix-construction)
     - [Cell 5: Latent Factor Decomposition (Truncated SVD)](#cell-5-latent-factor-decomposition-truncated-svd)
     - [Cell 6: Matrix Reconstruction & Predicted Preference Scores](#cell-6-matrix-reconstruction--predicted-preference-scores)
     - [Cell 7: Movie Recommendation Function with Input Validation](#cell-7-movie-recommendation-function-with-input-validation)
     - [Cell 8: Example User Recommendations](#cell-8-example-user-recommendations)
     - [Cell 9: Interactive Movie Similarity Demonstration with Validation](#cell-9-interactive-movie-similarity-demonstration-with-validation)
     - [Cell 10: Interactive User Recommendation Demonstration with Validation](#cell-10-interactive-user-recommendation-demonstration-with-validation)
     - [Cell 11: Error Evaluation (RMSE)](#cell-11-error-evaluation-rmse)
     - [Cell 12: Top-K Ranking Quality (Precision@K & Recall@K)](#cell-12-top-k-ranking-quality-precisionk--recallk)
     - [Cell 13: Visualizations: Baseline Recommendation Performance](#cell-13-visualizations-baseline-recommendation-performance)
     - [Cell 14: Final Baseline Metric Summary](#cell-14-final-baseline-metric-summary)
  3. [Part 1 Evaluation Metrics Reference](#3-part-1-evaluation-metrics-master-reference)
  4. [Part 1 Viva Voce Questions & Answers](#4-part-1-viva-voce-anticipated-questions--answers)
- **PART 2 — PROPOSED CAUSAL UPLIFT ENHANCEMENT**
  5. [Part 2 Conceptual Framing: From Preference to Causality](#5-part-2-conceptual-framing-from-preference-to-causality)
  6. [Part 2 Code & Theory Mapping (Cells 15–25)](#6-part-2-section-by-section-code--theory-mapping)
     - [Cell 15: Candidate Generation with Randomized User-Level Split](#cell-15-candidate-generation-from-svd-pool-with-randomized-user-split)
     - [Cell 16: Pre-Treatment Causal Feature Construction](#cell-16-pre-treatment-causal-feature-construction)
     - [Cell 17: Randomized Treatment Assignment Simulation](#cell-17-randomized-treatment-assignment-simulation)
     - [Cell 18: Engagement Outcome Construction (Potential Outcomes)](#cell-18-engagement-outcome-construction-potential-outcomes)
     - [Cell 19: Two-Model Causal Estimator with StandardScaler Pipeline](#cell-19-two-model-causal-estimator-t-learner-with-standardized-features)
     - [Cell 20: Individual Uplift Estimation](#cell-20-individual-uplift-estimation)
     - [Cell 21: Interactive Causal-Uplift Demonstration & Compact Comparison (User 1861)](#cell-21-interactive-causal-uplift-demonstration--compact-comparison-user-1861)
     - [Cell 22: Visual Demonstration: SVD Preference vs. Estimated Uplift](#cell-22-visual-demonstration-svd-preference-vs-estimated-uplift)
     - [Cell 23: Counterfactual Policy Evaluation with True Simulated Uplift](#cell-23-counterfactual-policy-evaluation-with-true-simulated-uplift)
     - [Cell 24: Comparative Policy Performance Scorecard](#cell-24-comparative-policy-performance-scorecard)
     - [Cell 25: Visualizations: Causal Uplift Diagnostics & True Incremental Lift Graph](#cell-25-visualizations-causal-uplift-diagnostics-and-true-incremental-lift-graph)
  7. [Part 2 Master Evaluation Metrics Reference](#7-part-2-master-evaluation-metrics-reference)
  8. [Part 2 Master Viva Voce Questions & Answers](#8-part-2-master-viva-voce-questions--answers)

---

### 1. Project Overview & Conceptual Framing

#### The Core Conventional Question
The conventional Netflix-style recommendation paradigm operates on an **observational preference-matching hypothesis**:
$$\text{"Given historical user ratings, which unseen movies is the user most likely to rate highly?"}$$

This baseline models **correlation and preference alignment**, ranking items by their predicted static preference score $\hat{r}_{ui}$.

#### The Dual Baseline Paradigms (Content-Based vs. Collaborative)
In Part 1, the system answers two distinct questions:
1. **Movie -> Similar Movies (Content-Based)**: "Given that a user likes *Toy Story* or *The Matrix*, which other movies share its genre profile?" (Measured via cosine similarity over multi-hot genre vectors).
2. **User -> Recommended Movies (Collaborative Filtering)**: "Given user 42's complete rating history across all movies, which unrated movies match their latent taste factors?" (Predicted via Truncated SVD).

#### Baseline vs. Part 2 (Causal AI / Uplift Preview)
- **Part 1 Baseline**: Predicts **organic engagement** $\hat{Y}_{ui} = P(\text{User } u \text{ likes Movie } i)$. It does not differentiate between movies the user would have watched anyway versus movies discovered *because* of the recommendation.
- **Part 2 Extension**: Applies **Causal Inference and Uplift Modeling** to estimate Individual Treatment Effects (ITE):
$$\tau_{ui} = Y_{ui}(1) - Y_{ui}(0)$$
answering: *"Which recommendation causes a positive change in user behavior that would not have happened otherwise?"*

---

### 2. Part 1 Section-by-Section Code & Theory Mapping

---

#### Cell 1: Dataset Ingestion

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

ratings = pd.read_csv('Neflix-Recommendation/ratings.dat', sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
movies = pd.read_csv('Neflix-Recommendation/movies.dat', sep='::', engine='python', names=['MovieID', 'Title', 'Genres'], encoding='latin-1')

print('Ratings shape:', ratings.shape)
print('Movies shape:', movies.shape)
display(ratings.head())
display(movies.head())
```

- **What each line does**: Imports required mathematical and modeling libraries (`pandas`, `numpy`, `matplotlib`, `scikit-learn`) and parses the raw MovieLens 1M `.dat` files into tabular DataFrames.
- **Why we use it**: MovieLens 1M uses double-colon (`::`) delimiters and latin-1 encoding for non-ASCII movie titles. Custom column headers ensure clean indexing.
- **Technical concept**: Ingestion and structured data schema representation.
- **Formulas / Schema**:
  - `ratings.dat`: UserID::MovieID::Rating::Timestamp
  - `movies.dat`: MovieID::Title::Genres
- **What the resulting value means**: 
  - `ratings`: 1,000,209 rows and 4 columns.
  - `movies`: 3,883 rows and 3 columns.
- **Viva answer**: *"We import scikit-learn's TruncatedSVD alongside standard data science libraries, and load the 1 million ratings and movie metadata using explicit column schemas and appropriate encodings."*

---

#### Cell 2: Statistics & Sparsity Computation

```python
num_users = ratings['UserID'].nunique()
num_movies = ratings['MovieID'].nunique()
num_ratings = len(ratings)

total_possible = num_users * num_movies
sparsity = (1.0 - (num_ratings / total_possible)) * 100.0

print(f'Unique users: {num_users:,}')
print(f'Unique movies: {num_movies:,}')
print(f'Total ratings: {num_ratings:,}')
print(f'Matrix Sparsity: {sparsity:.2f}%')
print(f'Average ratings per user: {num_ratings / num_users:.1f}')
print(f'Average ratings per movie: {num_ratings / num_movies:.1f}')
```

- **What each line does**: Computes unique user count ($6,040$), unique movie count ($3,706$), total interactions ($1,000,209$), and matrix sparsity ($95.53\%$).
- **Why we use it**: Recommendation systems face extreme data sparsity; quantifying sparsity explains why dimensionality reduction is necessary.
- **Technical concept**: Interaction density and rating sparsity.
- **Formula**:
$$\text{Sparsity (\%)} = \left(1 - \frac{|\mathcal{R}|}{|\mathcal{U}| \times |\mathcal{I}|}\right) \times 100$$
- **What the resulting value means**: $95.53\%$ of all possible user-movie combinations are unrated.
- **Viva answer**: *"The user-item matrix has 95.53% sparsity, meaning each user has rated only about 4.5% of the catalog on average. Latent factor models like SVD handle this by projecting sparse vectors into a dense, lower-dimensional space."*

---

#### Cell 3: User-Aware Train/Test Split

```python
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
train_ratings, test_ratings = train_test_split(
    ratings, test_size=0.20, random_state=RANDOM_SEED, stratify=ratings['UserID']
)

print(f'Training interactions: {len(train_ratings):,}')
print(f'Test interactions: {len(test_ratings):,}')
```

- **What each line does**: Sets random seed `42` and performs an 80/20 train/test split stratified on `UserID`.
- **Why we use it**: User stratification ensures every user appears in both train and test sets, avoiding the cold-start problem during evaluation.
- **Technical concept**: Stratified cross-sectional holdout validation.
- **What the resulting value means**: 800,167 interactions in training; 200,042 in test.
- **Viva answer**: *"We stratify by UserID so that every user's ratings are split 80% for training and 20% for testing. This guarantees we can personalize and evaluate recommendations for all users without unobserved cold-start users."*

---

#### Cell 4: User-Item Matrix Construction

```python
user_item_matrix = train_ratings.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)

print(f'User-Item matrix shape: {user_item_matrix.shape}')
display(user_item_matrix.iloc[:5, :8])
```

- **What each line does**: Reshapes the long interaction table into a 2D matrix ($6,040 \times 3,683$) and fills missing values with 0.
- **Why we use it**: Matrix factorization requires a complete numerical matrix.
- **Technical concept**: Sparse-to-dense matrix transformation. In collaborative filtering without bias terms, 0 denotes an unobserved item, not a 0-star rating.
- **Viva answer**: *"We pivot training ratings into a 6,040 by 3,683 user-item matrix. Unobserved ratings are filled with 0, representing missing interactions."*

---

#### Cell 5: Latent Factor Decomposition (Truncated SVD)

```python
svd = TruncatedSVD(n_components=20, random_state=RANDOM_SEED)
user_factors = svd.fit_transform(user_item_matrix)
item_factors = svd.components_

print(f'User latent factors shape: {user_factors.shape}')
print(f'Item latent factors shape: {item_factors.shape}')
print(f'Explained variance ratio sum: {svd.explained_variance_ratio_.sum():.4f}')
```

- **What each line does**: Decomposes the user-item matrix into 20 latent factor representations using Truncated SVD.
- **Why we use it**: Dimensionality reduction extracts dominant latent taste factors (e.g. action intensity, drama, humor) while filtering noise.
- **Technical concept**: Low-Rank Matrix Factorization.
- **Formula**:
$$\mathbf{R}_{m \times n} \approx \mathbf{U}_{m \times k} \mathbf{\Sigma}_{k \times k} \mathbf{V}_{k \times n}^T$$
where $k=20$, $\mathbf{P} = \mathbf{U}\mathbf{\Sigma}$ is the user factor matrix ($6,040 \times 20$), and $\mathbf{Q} = \mathbf{V}^T$ is the item factor matrix ($20 \times 3,683$).
- **What the resulting value means**: 20 components capture approximately $16.1\%$ of total variance while compressing data by over $99\%$.
- **Viva answer**: *"We use Truncated SVD with 20 latent factors. It compresses the rating space into 20 dimensions that represent underlying genres and preferences, computing in under two seconds."*

---

#### Cell 6: Matrix Reconstruction & Predicted Preference Scores

```python
predicted_ratings = svd.inverse_transform(user_factors)
predicted_df = pd.DataFrame(
    predicted_ratings,
    index=user_item_matrix.index,
    columns=user_item_matrix.columns
)

print(f'Predicted matrix shape: {predicted_df.shape}')
display(predicted_df.iloc[:5, :8].round(2))
```

- **What each line does**: Reconstructs the full rating matrix via the dot product of user and item factors: $\hat{\mathbf{R}} = \mathbf{P} \mathbf{Q}$.
- **Why we use it**: Provides a predicted preference score $\hat{r}_{ui}$ for every user-movie pair, including previously unobserved ones.
- **Technical concept**: Matrix reconstruction and score approximation.
- **Viva answer**: *"We reconstruct the matrix by projecting user factors back through the item components. The resulting score estimates how strongly a user's latent taste aligns with each movie's characteristics."*

---

#### Cell 7: Movie Recommendation Function with Input Validation

```python
def recommend_movies(user_id, n=10):
    if not isinstance(n, int) or n <= 0:
        print('n must be a positive integer.')
        return None
    if user_id not in user_item_matrix.index:
        print('User ID not found. Please enter a valid User ID.')
        return None
    user_rated = train_ratings[train_ratings['UserID'] == user_id]['MovieID'].values
    user_scores = predicted_df.loc[user_id].drop(labels=user_rated, errors='ignore')
    actual_n = min(n, len(user_scores))
    top_movies = user_scores.sort_values(ascending=False).head(actual_n)
    rec_df = pd.DataFrame({
        'Rank': range(1, actual_n + 1),
        'MovieID': top_movies.index,
        'Predicted Score': top_movies.values.round(3)
    }).merge(movies, on='MovieID', how='left')
    return rec_df[['Rank', 'MovieID', 'Title', 'Genres', 'Predicted Score']]
```

- **What each line does**: Validates that $n$ is a positive integer and that $user\_id$ exists; filters out movies already rated in training; handles requested $n$ larger than available candidates gracefully; and returns the top $n$ unseen movies sorted descending by predicted preference score.
- **Why we use it**: Recommending already-watched movies provides zero discovery value. Input validation prevents kernel tracebacks when invalid inputs or parameters are passed.
- **Technical concept**: Top-N Ranking with Candidate Exclusion and Error Handling.
- **Viva answer**: *"Our recommendation function validates user existence and parameter n, drops all movies already rated in training to ensure discovery of new items, and returns the top N ranked by predicted score."*

---

#### Cell 8: Example User Recommendations

```python
example_users = [1, 42, 100]

for uid in example_users:
    print(f'Top 10 Recommendations for User {uid}:')
    display(recommend_movies(user_id=uid, n=10))
    print()
```

- **What each line does**: Generates top-10 recommendation tables for Users 1, 42, and 100.
- **Why we use it**: Demonstrates personalized collaborative filtering across diverse taste cohorts:
  - **User 1**: Receives animated family classics (*Toy Story 2*, *The Lion King*, *Babe*, *The Little Mermaid*).
  - **User 42**: Receives iconic Sci-Fi and Action titles (*Star Trek: The Wrath of Khan*, *The Matrix*, *Star Trek IV*, *Stargate*, *Dune*).
  - **User 100**: Receives acclaimed dramas and thrillers (*American Beauty*, *Terminator 2*, *Alien*, *Pulp Fiction*).
- **Viva answer**: *"Testing distinct user IDs shows that the SVD latent factors capture distinct genre preferences without explicit genre rules."*

---

#### Cell 9: Interactive Movie Similarity Demonstration with Validation

```python
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movies['Genres'].str.split('|'))
genre_similarity = cosine_similarity(genre_matrix)

def get_similar_movies(title_query, top_n=5):
    if not isinstance(top_n, int) or top_n <= 0:
        print('top_n must be a positive integer.')
        return None
    query_clean = title_query.strip().lower()
    matches = movies[movies['Title'].str.lower() == query_clean]
    idx = None
    if len(matches) > 0:
        idx = matches.index[0]
    else:
        m = movies[movies['Title'].str.lower().str.contains(query_clean, regex=False)]
        if len(m) > 0:
            idx = m.index[0]
        else:
            for article in ['the ', 'a ', 'an ']:
                if query_clean.startswith(article):
                    core_name = query_clean[len(article):].split('(')[0].strip()
                    m2 = movies[movies['Title'].str.lower().str.contains(core_name, regex=False)]
                    if len(m2) > 0:
                        idx = m2.index[0]
                        break
            if idx is None:
                words = [w for w in query_clean.replace('(', '').replace(')', '').split() if w not in ['the', 'a', 'an']]
                if words:
                    m3 = movies[movies['Title'].str.lower().apply(lambda t: all(w in t for w in words))]
                    if len(m3) > 0:
                        idx = m3.index[0]
    if idx is None:
        print('Movie not found. Please enter a valid MovieLens movie title.')
        return None
    matched_title = movies.loc[idx, 'Title']
    scores = list(enumerate(genre_similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    other_scores = [s for i, s in scores if i != idx]
    actual_top_n = min(top_n, len(other_scores))
    top_indices = [i for i, s in scores if i != idx][:actual_top_n]
    top_scores = [s for i, s in scores if i != idx][:actual_top_n]
    result_df = pd.DataFrame({
        'Rank': range(1, actual_top_n + 1),
        'Movie': movies.loc[top_indices, 'Title'].values,
        'Genres': movies.loc[top_indices, 'Genres'].values,
        'Similarity': [round(s, 3) for s in top_scores]
    })
    print('PART 1 - MOVIE SIMILARITY DEMO')
    print(f'Input Movie: {matched_title}')
    print(f'Top {actual_top_n} Similar Movies:')
    return result_df

display(get_similar_movies('Toy Story', top_n=5))
print()
display(get_similar_movies('The Matrix (1999)', top_n=5))
print()
display(get_similar_movies('the matrix', top_n=5))
print()
get_similar_movies('Nonexistent Movie 999')
get_similar_movies('Toy Story', top_n=-1)
```

- **What each line does**:
  1. Converts movie genres into a binary multi-hot matrix using `MultiLabelBinarizer`.
  2. Calculates pairwise cosine similarity across all movies.
  3. Validates parameter `top_n` (rejecting non-positive or non-integer inputs).
  4. Provides a robust title search handling exact titles, case-insensitivity, leading articles (*The Matrix* -> *Matrix, The*), and token matches (*Toy Story* -> *Toy Story (1995)*).
  5. Returns the top 5 most similar movies excluding the query movie itself.
- **Why we use it**: Demonstrates **content-based movie similarity** ("If you watched X, here are movies like X"), which is fundamentally different from **collaborative user personalization**.
- **Technical concept**: Content-Based Filtering via Cosine Similarity over Multi-Hot Genre Vectors.
- **Formula**:
$$\text{Cosine Similarity}(\mathbf{g}_A, \mathbf{g}_B) = \frac{\mathbf{g}_A \cdot \mathbf{g}_B}{\|\mathbf{g}_A\| \|\mathbf{g}_B\|} = \frac{\sum_{k} g_{Ak} g_{Bk}}{\sqrt{\sum_k g_{Ak}^2} \sqrt{\sum_k g_{Bk}^2}}$$
where $\mathbf{g}_i \in \{0, 1\}^D$ is the binary genre vector for movie $i$.
- **Why the input movie is excluded**: The similarity of any movie with itself is identically $1.0$. Recommending the queried movie itself is trivial and unhelpful.
- **Viva answer**: *"We represent movie genres as multi-hot binary vectors and compute cosine similarity. For 'The Matrix (1999)', the system returns movies sharing identical Action|Sci-Fi|Thriller profiles like 'Terminator 2' and 'Johnny Mnemonic'. This is content similarity, which requires no user ratings, unlike SVD collaborative filtering."*

---

#### Cell 10: Interactive User Recommendation Demonstration with Validation

```python
demo_user = 42
print('PART 1 - PERSONALIZED SVD DEMO')
print(f'User ID: {demo_user}')
print('Top 10 Recommendations:')
display(recommend_movies(user_id=demo_user, n=10))
print()
recommend_movies(user_id=99999)
recommend_movies(user_id=42, n=0)
```

- **What each line does**: Demonstrates the interactive SVD recommendation pipeline for User 42 and tests boundary conditions: invalid user `99999` and invalid count `n=0`.
- **What the resulting output means**: User 42 receives personalized recommendations with scores ranging from 4.09 to 4.38; invalid inputs print descriptive messages without raising exceptions.
- **Viva answer**: *"This interactive demo demonstrates personalized SVD recommendations for User 42 while proving that our input validation prevents runtime crashes when given non-existent IDs or invalid parameter values."*

---

#### Cell 11: Error Evaluation (RMSE)

```python
valid_test = test_ratings[test_ratings['MovieID'].isin(predicted_df.columns)]

user_idx_map = {uid: i for i, uid in enumerate(predicted_df.index)}
movie_idx_map = {mid: i for i, mid in enumerate(predicted_df.columns)}

u_indices = [user_idx_map[uid] for uid in valid_test['UserID'].values]
m_indices = [movie_idx_map[mid] for mid in valid_test['MovieID'].values]

y_true = valid_test['Rating'].values
y_pred = predicted_df.values[u_indices, m_indices]

rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
print(f'Test RMSE: {rmse:.4f}')
```

- **What each line does**: Evaluates rating prediction error on held-out test ratings using vectorized NumPy array indexing.
- **Formula**:
$$\text{RMSE} = \sqrt{\frac{1}{|\mathcal{T}|} \sum_{(u, i) \in \mathcal{T}} (r_{ui} - \hat{r}_{ui})^2}$$
- **What the resulting value means**: Test RMSE is **2.7013**. In un-centered SVD with zero-imputation, missing entries shrink predictions toward zero, which shifts numerical predictions downward while preserving the relative ordering needed for ranking.
- **Viva answer**: *"Our test RMSE is 2.7013. Imputing unobserved ratings as 0 penalizes absolute rating values, but relative ranking quality is preserved for top-K recommendation."*

---

#### Cell 12: Top-K Ranking Quality (Precision@K & Recall@K)

```python
relevant_test = valid_test[valid_test['Rating'] >= 4].groupby('UserID')['MovieID'].apply(set).to_dict()
train_rated_dict = train_ratings.groupby('UserID')['MovieID'].apply(set).to_dict()
eval_users = [u for u in relevant_test if len(relevant_test[u]) > 0]

p5_list, r5_list, p10_list, r10_list = [], [], [], []

for u in eval_users:
    rel = relevant_test[u]
    train_rated = train_rated_dict.get(u, set())
    scores = predicted_df.loc[u].drop(labels=list(train_rated), errors='ignore')
    top10 = scores.nlargest(10).index
    top5 = top10[:5]
    
    hits5 = len(set(top5) & rel)
    hits10 = len(set(top10) & rel)
    
    p5_list.append(hits5 / 5)
    r5_list.append(hits5 / len(rel))
    p10_list.append(hits10 / 10)
    r10_list.append(hits10 / len(rel))

p5, r5 = np.mean(p5_list), np.mean(r5_list)
p10, r10 = np.mean(p10_list), np.mean(r10_list)

print('Baseline Recommendation Performance')
print('------------------------------------')
print(f'RMSE:         {rmse:.4f}')
print(f'Precision@5:  {p5:.4f}')
print(f'Recall@5:     {r5:.4f}')
print(f'Precision@10: {p10:.4f}')
print(f'Recall@10:    {r10:.4f}')
```

- **What each line does**: Defines relevant items as held-out test ratings $\ge 4.0$, computes Precision and Recall at cutoffs $K=5$ and $K=10$ per user, and averages across all evaluation users.
- **Formulas**:
$$\text{Precision@}K = \frac{|\text{Top-}K(u) \cap \mathcal{R}_{\text{test, rel}}(u)|}{K}$$
$$\text{Recall@}K = \frac{|\text{Top-}K(u) \cap \mathcal{R}_{\text{test, rel}}(u)|}{|\mathcal{R}_{\text{test, rel}}(u)|}$$
- **What the resulting values mean**:
  - **Precision@5**: $34.41\%$
  - **Recall@5**: $12.77\%$
  - **Precision@10**: $28.44\%$
  - **Recall@10**: $19.71\%$
- **Viva answer**: *"Precision@10 is 28.44%, meaning nearly 3 out of every 10 recommended movies are confirmed favorites (rated 4 or 5 stars) in the user's held-out test set."*

---

#### Cell 13: Visualizations: Baseline Recommendation Performance

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
...
```

- **What each line does**: Renders 4 diagnostic subplots: MovieLens rating distribution, top-10 most rated movies, top-10 SVD predictions for User 42, and Precision/Recall trade-off bar chart.
- **Viva answer**: *"These visualizations illustrate the rating distribution skewed toward 4 stars, catalog concentration on blockbusters, personalized predictions for User 42, and the classic Precision-Recall trade-off when expanding the recommendation list from 5 to 10 items."*

---

#### Cell 14: Final Baseline Metric Summary

```python
summary_df = pd.DataFrame({
    'Metric / Parameter': [
        'Number of users',
        'Number of movies',
        'Number of ratings',
        'SVD components',
        'RMSE',
        'Precision@5',
        'Recall@5',
        'Precision@10',
        'Recall@10'
    ],
    'Result': [
        f'{num_users:,}',
        f'{num_movies:,}',
        f'{num_ratings:,}',
        '20',
        f'{rmse:.4f}',
        f'{p5:.4f}',
        f'{r5:.4f}',
        f'{p10:.4f}',
        f'{r10:.4f}'
    ]
})

display(summary_df)
```

- **What each line does**: Compiles all core dataset statistics, model parameters, and baseline evaluation metrics into a clean reference table.
- **Viva answer**: *"This final table encapsulates our baseline benchmark: 6,040 users, 3,706 movies, 1M ratings, 20 latent factors, 2.70 RMSE, 28.4% Precision@10, and 19.7% Recall@10."*

---

### 3. Part 1 Evaluation Metrics Master Reference

| Metric | Formula | Meaning | Direction | Value in Notebook | Viva Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RMSE** | $\sqrt{\frac{1}{N}\sum (r - \hat{r})^2}$ | Prediction error magnitude | **Lower is better** | **2.7013** | *"Measures rating prediction accuracy on held-out test interactions."* |
| **Precision@5** | $\frac{|\text{Hits}@5|}{5}$ | Fraction of top 5 that are relevant ($\ge 4$) | **Higher is better** | **0.3441** | *"Over 34% of the top 5 recommendations are confirmed user favorites."* |
| **Recall@5** | $\frac{|\text{Hits}@5|}{|\text{Total Relevant}|}$ | Fraction of user favorites captured in top 5 | **Higher is better** | **0.1277** | *"Top 5 retrieves 12.8% of all relevant test movies."* |
| **Precision@10** | $\frac{|\text{Hits}@10|}{10}$ | Fraction of top 10 that are relevant ($\ge 4$) | **Higher is better** | **0.2844** | *"28.4% of top 10 recommendations are relevant."* |
| **Recall@10** | $\frac{|\text{Hits}@10|}{|\text{Total Relevant}|}$ | Fraction of user favorites captured in top 10 | **Higher is better** | **0.1971** | *"Expanding to 10 items captures nearly 20% of all relevant test movies."* |

---

### 4. Part 1 Viva Voce Anticipated Questions & Answers

#### Q1: Why did you use Truncated SVD instead of Standard SVD?
> **Answer**: *"Standard SVD requires computing all singular values and vectors, which is computationally expensive on a large matrix. Truncated SVD computes only the top $k=20$ dominant singular vectors directly via randomized algorithms, keeping execution to under two seconds."*

#### Q2: Why did you impute unobserved ratings with 0?
> **Answer**: *"In matrix factorization without explicit bias parameters, 0 represents an unobserved interaction rather than a negative review. While imputing 0 shifts reconstructed values downward (resulting in an RMSE of 2.70), the relative ordering of scores remains preserved for ranking."*

#### Q3: Why is Precision@K and Recall@K more important than RMSE for Netflix?
> **Answer**: *"A streaming platform displays a carousel of recommendations to users rather than predicted star values. Precision@K and Recall@K directly evaluate the quality of the items the user actually sees."*

#### Q4: How does Movie-to-Movie Content Similarity differ from SVD Collaborative Filtering?
> **Answer**: *"Movie similarity is based purely on movie metadata (multi-hot genres) using cosine similarity. It requires zero user interaction history and is ideal for 'More Like This' carousels. SVD collaborative filtering is personalized: it uses the collective rating history of all users to identify latent preference dimensions."*

---

# PART 2 — CAUSAL AI + UPLIFT MODELING

---

### 5. Part 2 Conceptual Framing: From Preference to Causality

#### The Fundamental Flaw of Conventional Recommenders
A conventional recommendation system (Part 1 SVD) optimizes for **observational engagement**:
$$\text{Objective: } \max \hat{r}_{ui}$$
This leads to recommending movies that users **already planned to watch anyway** (such as blockbuster sequels or ubiquitous cultural hits). While the observational accuracy of such recommendations appears high, their **incremental business value is zero**. 

#### The Causal Uplift Paradigm
Part 2 introduces **Uplift Modeling (Incremental Treatment Effect Estimation)**:
$$\text{Objective: } \max \tau(X_{ui}) = \mu_1(X_{ui}) - \mu_0(X_{ui})$$
where:
- $\mu_1(X_{ui}) = P(Y_{ui} = 1 \mid X_{ui}, W_{ui} = 1)$ is the probability of positive engagement **when recommended** (Treatment).
- $\mu_0(X_{ui}) = P(Y_{ui} = 1 \mid X_{ui}, W_{ui} = 0)$ is the probability of positive engagement **without recommendation** (Organic Control).
- $\tau(X_{ui})$ is the **Individual Treatment Effect (ITE) / Uplift**, representing the net increase in engagement caused *directly* by the recommendation.

#### Fair Comparison Protocol
To ensure a rigorous, scientifically valid comparison:
- Both SVD Baseline and Proposed Uplift policies compete over the **exact same pool of top-25 unseen candidates** generated by SVD.
- SVD ranks candidates by static predicted preference ($\hat{r}_{ui}$).
- Proposed Uplift reranks the same candidates by predicted incremental impact ($\hat{\tau}_{ui}$).
- Potential outcomes ($p_0, p_1, \tau_{\text{true}}$) are joined strictly post-hoc for evaluation and are never used as training features or ranking scores.

---

### 6. Part 2 Section-by-Section Code & Theory Mapping

---

#### Cell 15: Candidate Generation from SVD Pool with Randomized User Split

```python
candidate_pool_size = 25
train_rated_dict = train_ratings.groupby('UserID')['MovieID'].apply(set).to_dict()

all_causal_users = np.array(user_item_matrix.index)[:2000]
train_causal_users, test_causal_users = train_test_split(
    all_causal_users, test_size=0.20, random_state=RANDOM_SEED
)

print(f'Number of training users: {len(train_causal_users)}')
print(f'Number of evaluation users: {len(test_causal_users)}')
print(f'Confirmation that sets are disjoint: {len(set(train_causal_users) & set(test_causal_users)) == 0}')

def extract_candidates(users_list, pool_size=25):
    records = []
    for uid in users_list:
        rated = train_rated_dict.get(uid, set())
        scores = predicted_df.loc[uid].drop(labels=list(rated), errors='ignore')
        top_candidates = scores.nlargest(pool_size)
        for rank, (mid, score) in enumerate(top_candidates.items(), 1):
            records.append({
                'UserID': uid,
                'MovieID': mid,
                'svd_score': score,
                'candidate_rank': rank
            })
    return pd.DataFrame(records)

train_cand_df = extract_candidates(train_causal_users, candidate_pool_size)
eval_cand_df = extract_candidates(test_causal_users, candidate_pool_size)

print(f'Causal training candidates: {len(train_cand_df)}')
print(f'Causal evaluation candidates: {len(eval_cand_df)}')
display(train_cand_df.head())
```

- **What each line does**:
  1. Performs a reproducible 80/20 user-level randomized split using `train_test_split` with `random_state=42`.
  2. Confirms that training and evaluation user sets are completely disjoint (`len(set(train) & set(test)) == 0`).
  3. Extracts the top 25 unseen candidate movies for 1,600 training users (40,000 candidate pairs) and 400 evaluation users (10,000 candidate pairs).
- **Why we use it**: Splitting at the user level guarantees zero data leakage across users. Using the same candidate pool guarantees a strictly fair competition between preference ranking and causal ranking.
- **Viva answer**: *"We perform an 80/20 randomized split at the user level, confirming disjoint user sets. We extract the top 25 unseen movies per user from SVD so that both SVD and Uplift rerank the exact same items."*

---

#### Cell 16: Pre-Treatment Causal Feature Construction

```python
movies['ReleaseYear'] = movies['Title'].str.extract(r'\((\d{4})\)').astype(float).fillna(2000)
movies['GenreCount'] = movies['Genres'].apply(lambda x: len(x.split('|')))

user_train_count = train_ratings.groupby('UserID').size().to_dict()
user_train_avg = train_ratings.groupby('UserID')['Rating'].mean().to_dict()
movie_train_count = train_ratings.groupby('MovieID').size().to_dict()
movie_train_avg = train_ratings.groupby('MovieID')['Rating'].mean().to_dict()
movie_year_dict = movies.set_index('MovieID')['ReleaseYear'].to_dict()
movie_genre_dict = movies.set_index('MovieID')['GenreCount'].to_dict()

def attach_features(df):
    df_feat = df.copy()
    df_feat['user_train_count'] = df_feat['UserID'].map(user_train_count).fillna(0)
    df_feat['user_train_avg'] = df_feat['UserID'].map(user_train_avg).fillna(3.5)
    df_feat['movie_train_count'] = df_feat['MovieID'].map(movie_train_count).fillna(0)
    df_feat['movie_train_avg'] = df_feat['MovieID'].map(movie_train_avg).fillna(3.5)
    df_feat['release_year'] = df_feat['MovieID'].map(movie_year_dict).fillna(2000)
    df_feat['genre_count'] = df_feat['MovieID'].map(movie_genre_dict).fillna(1)
    return df_feat

train_feat_df = attach_features(train_cand_df)
eval_feat_df = attach_features(eval_cand_df)

feature_cols = [
    'user_train_count', 'user_train_avg', 'movie_train_count',
    'movie_train_avg', 'release_year', 'genre_count',
    'svd_score', 'candidate_rank'
]

display(train_feat_df[feature_cols].head())
```

- **What each line does**: Constructs 8 pre-treatment user, movie, and recommender features strictly from historical training interactions and metadata.
- **Features Extracted**:
  1. `user_train_count`: Total ratings by the user in training (activity level).
  2. `user_train_avg`: Average rating given by the user (rating bias).
  3. `movie_train_count`: Number of ratings received by the movie (popularity).
  4. `movie_train_avg`: Average rating of the movie (quality).
  5. `release_year`: Extracted release year.
  6. `genre_count`: Number of assigned genres (niche vs. cross-genre).
  7. `svd_score`: Baseline predicted preference score.
  8. `candidate_rank`: Rank position (1 to 25) within SVD pool.
- **Viva answer**: *"We construct 8 pre-treatment features capturing user activity, movie popularity, release year, and baseline SVD score strictly from training data to avoid any future leakage."*

---

#### Cell 17: Randomized Treatment Assignment Simulation

```python
np.random.seed(RANDOM_SEED)
train_feat_df['W'] = np.random.binomial(1, 0.5, size=len(train_feat_df))

print('Treatment distribution:')
print(train_feat_df['W'].value_counts(normalize=True))
```

- **What each line does**: Simulates recommendation exposure $W \in \{0, 1\}$ using a 50/50 randomized A/B test design.
- **Technical concept**: Randomized Controlled Trial (RCT) simulation.
- **Formula**:
$$W_{ui} \sim \text{Bernoulli}(p=0.50)$$
- **Viva answer**: *"Because MovieLens contains observed ratings rather than live Netflix impression logs, we simulate an A/B test with 50/50 randomized treatment assignment. This satisfies the unconfoundedness assumption required for unbiased causal estimation."*

---

#### Cell 18: Engagement Outcome Construction (Potential Outcomes)

```python
from sklearn.preprocessing import StandardScaler

scaler_dgp = StandardScaler()
X_tr_std = scaler_dgp.fit_transform(train_feat_df[feature_cols])

z0_tr = -0.8 + 0.35 * X_tr_std[:, 2] + 0.25 * X_tr_std[:, 6] + 0.15 * X_tr_std[:, 3]
p0_tr = 1 / (1 + np.exp(-z0_tr))
z1_tr = -0.4 + 0.10 * X_tr_std[:, 2] + 0.50 * X_tr_std[:, 6] + 0.30 * X_tr_std[:, 3] - 0.20 * X_tr_std[:, 7]
p1_tr = 1 / (1 + np.exp(-z1_tr))

prob_train = np.where(train_feat_df['W'] == 1, p1_tr, p0_tr)
train_feat_df['Y'] = np.random.binomial(1, prob_train)

print(f'Overall positive engagement rate: {train_feat_df["Y"].mean():.4f}')
print(f'Treatment group positive rate: {train_feat_df[train_feat_df["W"] == 1]["Y"].mean():.4f}')
print(f'Control group positive rate: {train_feat_df[train_feat_df["W"] == 0]["Y"].mean():.4f}')
```

- **What each line does**: Standardizes pre-treatment features and models potential outcome probabilities:
  - Organic control probability $p_0(X) = \sigma(z_0(X))$: Driven by high movie popularity (`movie_train_count`, index 2) and baseline preference (`svd_score`, index 6).
  - Treatment response probability $p_1(X) = \sigma(z_1(X))$: Adds recommendation discovery lift, where recommendation exposure provides the largest boost on high-preference items that are not already saturated by popularity.
  - Realized binary outcome $Y \sim \text{Bernoulli}(p)$: Realizes $Y = W \cdot Y(1) + (1-W) \cdot Y(0)$.
- **Technical concept**: Neyman-Rubin Potential Outcomes Framework.
- **Viva answer**: *"We model organic control response p0 as driven by popularity and baseline preference, while treatment response p1 adds recommendation discovery lift. The realized outcome Y is simulated according to treatment assignment W."*

---

#### Cell 19: Two-Model Causal Estimator (T-Learner) with Standardized Features

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model_control = make_pipeline(StandardScaler(), LogisticRegression(random_state=RANDOM_SEED, max_iter=500))
model_treatment = make_pipeline(StandardScaler(), LogisticRegression(random_state=RANDOM_SEED, max_iter=500))

ctrl_data = train_feat_df[train_feat_df['W'] == 0]
treat_data = train_feat_df[train_feat_df['W'] == 1]

model_control.fit(ctrl_data[feature_cols], ctrl_data['Y'])
model_treatment.fit(treat_data[feature_cols], treat_data['Y'])

print('Models trained successfully with standardized features.')
print(f'Model 0 (Control) intercept: {model_control.named_steps["logisticregression"].intercept_[0]:.4f}')
print(f'Model 1 (Treatment) intercept: {model_treatment.named_steps["logisticregression"].intercept_[0]:.4f}')
```

- **What each line does**: Encapsulates `StandardScaler` and `LogisticRegression` into a scikit-learn pipeline for both control and treatment models. Model 0 is fitted on control data ($W=0$), and Model 1 is fitted on treatment data ($W=1$).
- **Why we standardize**: Standardization ensures features with large numerical magnitudes (e.g., `movie_train_count` up to 3,000) do not disproportionately dominate regularized logistic regression weights relative to smaller-scale features (e.g., `genre_count`).
- **Technical concept**: T-Learner (Two-Model Meta-Learner) with Input Standardization.
- **Formulas**:
$$\hat{\mu}_0(X) = P(Y=1 \mid X, W=0) = \sigma(\mathbf{w}_0^T \tilde{\mathbf{X}} + b_0)$$
$$\hat{\mu}_1(X) = P(Y=1 \mid X, W=1) = \sigma(\mathbf{w}_1^T \tilde{\mathbf{X}} + b_1)$$
where $\tilde{\mathbf{X}}$ represents z-score standardized features.
- **Viva answer**: *"We use a T-Learner consisting of two logistic regression models trained inside a StandardScaler pipeline. Model 0 learns organic response from unrecommended pairs, and Model 1 learns response under recommendation exposure."*

---

#### Cell 20: Individual Uplift Estimation

```python
eval_feat_df['mu0'] = model_control.predict_proba(eval_feat_df[feature_cols])[:, 1]
eval_feat_df['mu1'] = model_treatment.predict_proba(eval_feat_df[feature_cols])[:, 1]
eval_feat_df['uplift'] = eval_feat_df['mu1'] - eval_feat_df['mu0']

print('Estimated Uplift Statistics:')
display(eval_feat_df[['mu0', 'mu1', 'uplift']].describe().round(4))
```

- **What each line does**: Evaluates Model 0 and Model 1 on the 10,000 evaluation candidate pairs (400 users $\times$ 25 movies) and computes estimated uplift $\hat{\tau}(X) = \hat{\mu}_1(X) - \hat{\mu}_0(X)$.
- **Formula**:
$$\hat{\tau}(X_{ui}) = \hat{\mu}_1(X_{ui}) - \hat{\mu}_0(X_{ui})$$
- **What the resulting value means**: Positive values indicate that recommendation exposure is predicted to cause a net increase in the probability of engagement.
- **Viva answer**: *"Estimated uplift is the difference between predicted treatment response and predicted control response. It estimates the incremental gain in engagement caused directly by showing the recommendation."*

---

#### Cell 21: Interactive Causal-Uplift Demonstration & Compact Comparison (User 1861)

```python
demo_uid = 1861
user_demo = eval_feat_df[eval_feat_df['UserID'] == demo_uid].merge(movies[['MovieID', 'Title', 'Genres']], on='MovieID')

svd_top5 = user_demo.sort_values('svd_score', ascending=False).head(5).copy()
svd_top5['Rank'] = range(1, 6)

upl_top5 = user_demo.sort_values('uplift', ascending=False).head(5).copy()
upl_top5['Rank'] = range(1, 6)

print(f'PART 2 - OUR CAUSAL-UPLIFT DEMO (User {demo_uid})')
print('\nTRADITIONAL SVD')
display(svd_top5[['Rank', 'Title', 'svd_score']].rename(columns={'Title': 'Movie', 'svd_score': 'SVD Score'}).round(3))

print('\nOUR CAUSAL-UPLIFT METHOD')
display(upl_top5[['Rank', 'Title', 'uplift']].rename(columns={'Title': 'Movie', 'uplift': 'Estimated Uplift'}).round(4))

top_svd_ids = set(svd_top5['MovieID'])
top_upl_ids = set(upl_top5['MovieID'])
union_ids = top_svd_ids | top_upl_ids

comp_df = user_demo[user_demo['MovieID'].isin(union_ids)].copy()
comp_df['Selected by SVD'] = comp_df['MovieID'].isin(top_svd_ids).map({True: 'Yes', False: 'No'})
comp_df['Selected by Uplift'] = comp_df['MovieID'].isin(top_upl_ids).map({True: 'Yes', False: 'No'})
comp_table = comp_df.sort_values('uplift', ascending=False)[['Title', 'svd_score', 'uplift', 'Selected by SVD', 'Selected by Uplift']].reset_index(drop=True)
comp_table.columns = ['Movie', 'SVD Preference', 'Estimated Uplift', 'Selected by SVD', 'Selected by Uplift']
comp_table['SVD Preference'] = comp_table['SVD Preference'].round(3)
comp_table['Estimated Uplift'] = comp_table['Estimated Uplift'].round(4)

print(f'\nCOMPACT DEMONSTRATION COMPARISON TABLE (User {demo_uid})')
display(comp_table)
```

- **What each line does**: Evaluates Demo User 1861 from the held-out evaluation set over the identical 25-candidate pool, producing SVD top 5, Uplift top 5, and a compact comparison table.
- **The Core Viva Result**:

| Movie | SVD Preference | Estimated Uplift | Selected by SVD | Selected by Uplift | Viva Story |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **American Graffiti (1973)** | 3.175 | **0.2932** | **Yes** | **Yes** | High preference AND highest uplift |
| **One Flew Over the Cuckoo's Nest (1975)** | 3.444 | **0.2704** | **Yes** | **Yes** | Strong preference and strong incrementality |
| **Bridge on the River Kwai, The (1957)** | 2.778 | **0.2682** | No | **Yes** | Tailored discovery gem promoted by uplift |
| **2001: A Space Odyssey (1968)** | 3.287 | **0.2630** | **Yes** | **Yes** | Balanced high performer |
| **Dances with Wolves (1990)** | 3.068 | **0.2433** | No | **Yes** | High discovery impact promoted by uplift |
| **Back to the Future (1985)** | **3.474** | 0.1863 | **Yes** | No | Top SVD pick dropped due to organic saturation |
| **Saving Private Ryan (1998)** | 3.229 | 0.1803 | **Yes** | No | High baseline viewership dropped by uplift |

- **Viva answer**: *"This table proves the central thesis of our project: 'Back to the Future' has the highest SVD preference (3.474), but because it is a ubiquitous blockbuster that the user would watch organically anyway, its estimated uplift is lower (0.1863). The uplift model de-selects it and replaces it with 'The Bridge on the River Kwai' (+0.2682), which produces higher incremental response."*

---

#### Cell 22: Visual Demonstration: SVD Preference vs. Estimated Uplift

```python
fig, ax1 = plt.subplots(figsize=(10, 5))
...
```

- **What each line does**: Generates a dual-axis horizontal grouped bar chart for Demo User 1861 comparing SVD Preference Score vs. Estimated Incremental Uplift.
- **Viva answer**: *"This chart visually contrasts 'What the user likes' against 'What recommendation causes behavior change,' making the divergence between static preference and causal lift obvious."*

---

#### Cell 23: Counterfactual Policy Evaluation with True Simulated Uplift

```python
X_ev_std = scaler_dgp.transform(eval_feat_df[feature_cols])
z0_ev = -0.8 + 0.35 * X_ev_std[:, 2] + 0.25 * X_ev_std[:, 6] + 0.15 * X_ev_std[:, 3]
p0_ev = 1 / (1 + np.exp(-z0_ev))
z1_ev = -0.4 + 0.10 * X_ev_std[:, 2] + 0.50 * X_ev_std[:, 6] + 0.30 * X_ev_std[:, 3] - 0.20 * X_ev_std[:, 7]
p1_ev = 1 / (1 + np.exp(-z1_ev))

true_eval_df = pd.DataFrame({
    'UserID': eval_feat_df['UserID'],
    'MovieID': eval_feat_df['MovieID'],
    'true_p0': p0_ev,
    'true_p1': p1_ev,
    'true_uplift': p1_ev - p0_ev
})

relevant_test_dict = test_ratings[test_ratings['Rating'] >= 4].groupby('UserID')['MovieID'].apply(set).to_dict()

svd_p10_list, svd_r10_list = [], []
svd_resp_list, svd_est_upl_list, svd_true_upl_list = [], [], []

upl_p10_list, upl_r10_list = [], []
upl_resp_list, upl_est_upl_list, upl_true_upl_list = [], [], []

svd_unique_recs = set()
upl_unique_recs = set()
overlap_list = []

for uid, group in eval_feat_df.groupby('UserID'):
    rel_movies = relevant_test_dict.get(uid, set())
    
    svd_top10 = group.sort_values('svd_score', ascending=False).head(10)
    svd_set = set(svd_top10['MovieID'])
    svd_unique_recs.update(svd_set)
    svd_hits = len(svd_set & rel_movies)
    svd_p10_list.append(svd_hits / 10)
    svd_r10_list.append(svd_hits / len(rel_movies) if len(rel_movies) > 0 else 0)
    svd_resp_list.append(svd_top10['mu1'].mean())
    svd_est_upl_list.append(svd_top10['uplift'].mean())
    
    upl_top10 = group.sort_values('uplift', ascending=False).head(10)
    upl_set = set(upl_top10['MovieID'])
    upl_unique_recs.update(upl_set)
    upl_hits = len(upl_set & rel_movies)
    upl_p10_list.append(upl_hits / 10)
    upl_r10_list.append(upl_hits / len(rel_movies) if len(rel_movies) > 0 else 0)
    upl_resp_list.append(upl_top10['mu1'].mean())
    upl_est_upl_list.append(upl_top10['uplift'].mean())
    
    svd_eval_joined = svd_top10.merge(true_eval_df, on=['UserID', 'MovieID'])
    upl_eval_joined = upl_top10.merge(true_eval_df, on=['UserID', 'MovieID'])
    
    svd_true_upl_list.append(svd_eval_joined['true_uplift'].mean())
    upl_true_upl_list.append(upl_eval_joined['true_uplift'].mean())
    
    overlap_list.append(len(svd_set & upl_set))

total_catalog_size = ratings['MovieID'].nunique()
svd_coverage = (len(svd_unique_recs) / total_catalog_size) * 100
upl_coverage = (len(upl_unique_recs) / total_catalog_size) * 100

print(f'Mean top-10 overlap: {np.mean(overlap_list):.2f} / 10 movies')
```

- **What each line does**:
  1. Computes the simulator's true potential outcomes ($p_0, p_1, \tau_{\text{true}} = p_1 - p_0$) on evaluation candidates.
  2. Generates top-10 recommendations for SVD (using `svd_score`) and Uplift (using estimated `uplift`) **independently and without knowledge of true potential outcomes**.
  3. Joins the true simulated values strictly post-hoc to calculate:
     - `Mean Predicted Treatment Response`: $\frac{1}{K} \sum \hat{\mu}_1(X)$
     - `Mean Estimated Uplift@10`: $\frac{1}{K} \sum \hat{\tau}(X)$
     - `Mean True Simulated Uplift@10`: $\frac{1}{K} \sum \tau_{\text{true}}(X)$
- **Why post-hoc joining is mandatory**: Guarantees zero data leakage; true simulated values evaluate the policy without influencing recommendation rankings.
- **Viva answer**: *"We generate recommendations using only predicted SVD preference and estimated model uplift. We then join true simulated uplift post-hoc to independently verify whether the proposed policy actually achieved higher incremental impact."*

---

#### Cell 24: Comparative Policy Performance Scorecard

```python
comparison_df = pd.DataFrame({
    'Metric': [
        'Precision@10',
        'Recall@10',
        'Mean Predicted Treatment Response',
        'Mean Estimated Uplift@10',
        'Mean True Simulated Uplift@10',
        'Catalog Coverage',
        'Ranking Objective'
    ],
    'SVD Baseline': [
        f'{np.mean(svd_p10_list):.4f}',
        f'{np.mean(svd_r10_list):.4f}',
        f'{np.mean(svd_resp_list):.4f}',
        f'{np.mean(svd_est_upl_list):.4f}',
        f'{np.mean(svd_true_upl_list):.4f}',
        f'{svd_coverage:.2f}%',
        'Predicted preference'
    ],
    'Proposed Uplift': [
        f'{np.mean(upl_p10_list):.4f}',
        f'{np.mean(upl_r10_list):.4f}',
        f'{np.mean(upl_resp_list):.4f}',
        f'{np.mean(upl_est_upl_list):.4f}',
        f'{np.mean(upl_true_upl_list):.4f}',
        f'{upl_coverage:.2f}%',
        'Predicted incremental response'
    ]
})

display(comparison_df)
```

- **Scorecard Results**:

| Metric | SVD Baseline | Proposed Uplift | Interpretation |
| :--- | :--- | :--- | :--- |
| **Precision@10** | **0.2775** | 0.2725 | Slight trade-off (-0.5%) to explore discovery items |
| **Recall@10** | **0.1966** | 0.1933 | Retains 98.3% of baseline retrieval coverage |
| **Mean Predicted Treatment Response** | **0.5220** | 0.5042 | Gross probability of positive engagement under exposure |
| **Mean Estimated Uplift@10** | 0.1467 | **0.1538** | **+4.8% relative gain** in model-estimated incremental impact |
| **Mean True Simulated Uplift@10** | 0.1480 | **0.1576** | **+6.5% relative gain** in ground-truth causal lift |
| **Catalog Coverage** | 10.20% | **11.12%** | **+9.0% broader catalog exploration** |
| **Ranking Objective** | Predicted preference | Predicted incremental response | Observational correlation vs. Causal intervention |

- **Viva answer**: *"Our primary metric, Mean True Simulated Uplift@10, increases from 0.1480 under SVD to 0.1576 under Proposed Uplift. This confirms that our uplift layer successfully selects movies with greater incremental causal impact while maintaining 27.25% Precision@10."*

---

#### Cell 25: Visualizations: Causal Uplift Diagnostics and True Incremental Lift Graph

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
...
```

- **Visualizations Included**:
  1. `axes[0, 0]`: SVD Preference Score vs. Estimated Uplift (Scatter Plot) — illustrates non-linear divergence.
  2. `axes[0, 1]`: **True Incremental Lift@10: SVD vs Proposed Uplift** (Primary Comparison Bar Chart) — displays Mean True Uplift values (0.1480 vs 0.1576) with data labels.
  3. `axes[1, 0]`: Precision@10 & Recall@10 Comparison — demonstrates near-parity in relevance metrics.
  4. `axes[1, 1]`: Top-10 Recommendation Overlap Distribution — displays user overlap distribution with mean overlap of 5.86 movies.
- **Viva answer**: *"The upper-right chart directly answers our research question: Proposed Uplift achieves higher true incremental lift (0.1576 vs 0.1480). The bottom-right histogram shows that the systems agree on about 6 movies per user, while diverging on the remaining 4 to optimize for incrementality."*

---

### 7. Part 2 Master Evaluation Metrics Reference

| Metric | Mathematical Definition | Role & Purpose | Comparison (SVD vs. Uplift) | Viva Voce Explanation |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Predicted Treatment Response** | $\frac{1}{N} \sum_{u} \frac{1}{K} \sum_{i \in \text{Top-}K} \hat{\mu}_1(X_{ui})$ | Average estimated probability that the user engages positively when recommended. | **0.5220 vs. 0.5042** | *"Gross probability of engagement when shown. SVD is slightly higher because it picks popular blockbusters that users watch anyway."* |
| **Mean Estimated Uplift@10** | $\frac{1}{N} \sum_{u} \frac{1}{K} \sum_{i \in \text{Top-}K} \hat{\tau}(X_{ui})$ | Average model-predicted incremental uplift $\hat{\mu}_1 - \hat{\mu}_0$ among top 10 items. | **0.1467 vs. 0.1538** | *"Model-predicted incrementality. Uplift selects items predicted to produce higher net impact."* |
| **Mean True Simulated Uplift@10** | $\frac{1}{N} \sum_{u} \frac{1}{K} \sum_{i \in \text{Top-}K} \tau_{\text{true}}(X_{ui})$ | Average simulator ground-truth incremental effect $p_1 - p_0$ among top 10 items. | **0.1480 vs. 0.1576** | *"Primary evaluation metric. Confirms that our uplift model successfully selects items that generate greater true incremental response."* |
| **Catalog Coverage** | $\frac{|\bigcup_{u} \text{Top-}K(u)|}{|\mathcal{I}|} \times 100$ | Percentage of unique catalog movies recommended across all test users. | **10.20% vs. 11.12%** | *"Uplift diversifies recommendations, promoting a wider variety of catalog titles beyond popular blockbusters."* |
| **Top-10 Overlap** | $|\text{Top-}K_{\text{SVD}}(u) \cap \text{Top-}K_{\text{Uplift}}(u)|$ | Number of shared titles between SVD and Uplift recommendations. | **5.86 / 10 movies** | *"The models share about 6 core high-preference titles, but differ on 4 items where uplift swaps organic blockbusters for discovery items."* |

---

### 8. Part 2 Master Viva Voce Questions & Answers

#### Q1: Why did you standardize features before fitting Logistic Regression?
> **Answer**: *"Features like `movie_train_count` range into the thousands, whereas `genre_count` ranges from 1 to 6. Without standardization, the L2 regularization penalty in LogisticRegression disproportionately shrinks coefficients of smaller-scale features. Using a StandardScaler pipeline puts all features on zero mean and unit variance, ensuring fair model fitting."*

#### Q2: Why did you split users randomly instead of splitting interactions?
> **Answer**: *"If we split interactions randomly, candidate pairs for the same user would appear in both training and evaluation sets, leading to user-level data leakage. Splitting at the user level guarantees that the 400 evaluation users are completely unseen by the causal models."*

#### Q3: What is the difference between Estimated Uplift and True Simulated Uplift?
> **Answer**: *"Estimated Uplift ($\hat{\tau}$) is the prediction made by our trained T-Learner models ($\hat{\mu}_1 - \hat{\mu}_0$). True Simulated Uplift ($\tau_{\text{true}}$) is the simulator's known ground-truth effect ($p_1 - p_0$). True uplift is joined strictly post-hoc to independently verify whether the model's ranking actually maximized causal impact."*

#### Q4: Why is Mean Predicted Treatment Response higher for SVD, but True Uplift higher for Proposed Uplift?
> **Answer**: *"SVD selects popular movies with high gross engagement ($\mu_1 \approx 0.52$), but their organic baseline ($\mu_0$) is also very high because users would watch them without a recommendation. Proposed Uplift targets items where $\mu_1 - \mu_0$ is maximized, achieving greater net incremental gain (+0.1576 vs +0.1480)."*

#### Q5: Did you use ground-truth causal outcomes when generating recommendations?
> **Answer**: *"No. Recommendations were generated strictly using predicted SVD preference scores and estimated uplift from the trained models. The simulator's true potential outcomes were joined only after recommendation lists were finalized to evaluate performance without leakage."*
