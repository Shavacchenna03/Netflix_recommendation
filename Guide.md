# Netflix Recommendation Research Lab
## Complete Theory, Code Mapping, and Viva Voce Master Guide (Part 1 & Part 2)

---

### Master Table of Contents
- **PART 1 — NETFLIX-STYLE BASELINE RECOMMENDATION**
  1. [Project Overview & Conceptual Framing](#1-project-overview--conceptual-framing)
  2. [Part 1 Code & Theory Mapping (Cells 1–8)](#2-part-1-section-by-section-code--theory-mapping)
     - [Cell 1: Environment Configuration & Library Ingestion](#cell-1-environment-configuration--library-ingestion)
     - [Cell 2: Dataset Loading & Verification](#cell-2-dataset-loading--verification)
     - [Cell 3: Interaction Matrix Sparsity & Stratified Train/Test Split](#cell-3-interaction-matrix-sparsity--stratified-traintest-split)
     - [Cell 4: Latent Factor Decomposition & Score Reconstruction](#cell-4-latent-factor-decomposition--score-reconstruction)
     - [Cell 5: Recommendation Function Definition with Input Validation](#cell-5-recommendation-function-definition-with-input-validation)
     - [Cell 6: Personalized SVD Recommendation Demonstration](#cell-6-personalized-svd-recommendation-demonstration)
     - [Cell 7: Content-Based Movie Similarity Demonstration with Validation](#cell-7-content-based-movie-similarity-demonstration-with-validation)
     - [Cell 8: Baseline Accuracy & Retrieval Metrics (RMSE, Precision@K, Recall@K)](#cell-8-baseline-accuracy--retrieval-metrics-rmse-precisionk-recallk)
  3. [Part 1 Evaluation Metrics Reference](#3-part-1-evaluation-metrics-master-reference)
  4. [Part 1 Viva Voce Questions & Answers](#4-part-1-viva-voce-anticipated-questions--answers)
- **PART 2 — PROPOSED CAUSAL AI + UPLIFT RECOMMENDATION**
  5. [Part 2 Conceptual Framing: From Preference to Causality](#5-part-2-conceptual-framing-from-preference-to-causality)
  6. [Part 2 Code & Theory Mapping (Cells 9–16)](#6-part-2-section-by-section-code--theory-mapping)
     - [Cell 9: Randomized User Split & Candidate Generation from SVD Pool](#cell-9-randomized-user-split--candidate-generation-from-svd-pool)
     - [Cell 10: Randomized Treatment Assignment & Potential Outcomes Simulation](#cell-10-randomized-treatment-assignment--potential-outcomes-simulation)
     - [Cell 11: Two-Model Causal Estimator (T-Learner) & Individual Uplift Estimation](#cell-11-two-model-causal-estimator-t-learner--individual-uplift-estimation)
     - [Cell 12: Interactive Causal-Uplift Demonstration & Compact Comparison (User 1861)](#cell-12-interactive-causal-uplift-demonstration--compact-comparison-user-1861)
     - [Cell 13: Visual Demonstration: SVD Preference vs. Estimated Uplift](#cell-13-visual-demonstration-svd-preference-vs-estimated-uplift)
     - [Cell 14: Counterfactual Policy Evaluation with True Simulated Uplift](#cell-14-counterfactual-policy-evaluation-with-true-simulated-uplift)
     - [Cell 15: Comparative Policy Performance Scorecard](#cell-15-comparative-policy-performance-scorecard)
     - [Cell 16: Primary Visualizations: True Incremental Lift@10 & Diagnostics](#cell-16-primary-visualizations-true-incremental-lift10--diagnostics)
  7. [Part 2 Master Evaluation Metrics Reference](#7-part-2-master-evaluation-metrics-reference)
  8. [Part 2 Master Viva Voce Questions & Answers](#8-part-2-master-viva-voce-questions--answers)

---

### 1. Project Overview & Conceptual Framing

#### The Core Conventional Question
The conventional Netflix-style recommendation paradigm operates on an **observational preference-matching hypothesis**:
$$\text{"Given historical user ratings, which unseen movies is the user most likely to prefer?"}$$

This baseline models **correlation and preference alignment**, ranking items by their predicted static preference score $\hat{r}_{ui}$.

#### Important Terminology Distinction
- **What this project is**: An academic research lab implementing a **Netflix-style baseline recommendation system** using publicly available MovieLens 1M benchmark data.
- **What this project is NOT**: We do not claim to reproduce Netflix's proprietary production algorithm or live streaming infrastructure.
- **The Research Claim**: *"Our proposed causal uplift decision layer improves incremental recommendation targeting over the conventional SVD baseline under a controlled simulated experimental setting."*

#### The Dual Baseline Paradigms (Content-Based vs. Collaborative)
In Part 1, the system answers two distinct questions:
1. **Movie -> Similar Movies (Content-Based)**: "Given that a user likes *Toy Story* or *The Matrix*, which other movies share its genre profile?" (Measured via cosine similarity over multi-hot genre vectors).
2. **User -> Recommended Movies (Collaborative Filtering)**: "Given user 42's complete rating history across all movies, which unrated movies match their latent taste factors?" (Predicted via Truncated SVD).

#### Baseline vs. Part 2 (Causal AI / Uplift Preview)
- **Part 1 Baseline**: Predicts **gross engagement** $\hat{Y}_{ui} = P(\text{User } u \text{ likes Movie } i)$. It does not differentiate between movies the user would have watched anyway versus movies discovered *because* of the recommendation.
- **Part 2 Extension**: Applies **Causal Inference and Uplift Modeling** to estimate Individual Treatment Effects (ITE):
$$\tau_{ui} = Y_{ui}(1) - Y_{ui}(0)$$
answering: *"Which recommendation causes an incremental change in user behavior that would not have happened organically?"*

---

### 2. Part 1 Section-by-Section Code & Theory Mapping

---

#### Cell 1: Environment Configuration & Library Ingestion

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
```

- **Code line / operation**: Imports core analytical packages from pandas, NumPy, scikit-learn, and matplotlib, and fixes the global random seed `RANDOM_SEED = 42`.
- **Why it is used**: Provides standard, defensible tools for tabular data manipulation, low-rank matrix decomposition, cosine distance computation, feature scaling, and regularized logistic classification. Fixing the seed guarantees exact numerical reproducibility.
- **Theory**: Reproducibility in empirical machine learning ensures that train/test splits, SVD singular vectors, randomized treatment assignment, and logistic regression solver iterations are deterministic across different kernel runs.
- **Viva question**: *"Why did you set a random seed of 42?"*
  - **Answer**: *"To ensure complete numerical reproducibility across randomized train/test splits, SVD randomized solvers, and simulated treatment assignment."*

---

#### Cell 2: Dataset Loading & Verification

```python
ratings = pd.read_csv('Neflix-Recommendation/ratings.dat', sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
movies = pd.read_csv('Neflix-Recommendation/movies.dat', sep='::', engine='python', names=['MovieID', 'Title', 'Genres'], encoding='latin-1')

print('Ratings shape:', ratings.shape)
print('Movies shape:', movies.shape)
display(ratings.head())
display(movies.head())
```

- **Code line / operation**: Parses raw MovieLens 1M `.dat` files into pandas DataFrames using custom double-colon delimiters (`::`), explicit column names, and `latin-1` character encoding.
- **Why it is used**: MovieLens uses `::` delimiters and non-ASCII character encoding for European movie titles. Explicit schemas ensure proper data ingestion without index corruption.
- **Theory**: Structured tabular data schemas:
  - `ratings`: UserID::MovieID::Rating::Timestamp ($1,000,209$ rows $\times$ 4 columns)
  - `movies`: MovieID::Title::Genres ($3,883$ rows $\times$ 3 columns)
- **Output interpretation**: Confirms that all 1 million ratings and nearly 3,900 movies are successfully parsed into memory with valid headers.
- **Viva question**: *"What dataset are you using and why?"*
  - **Answer**: *"We use the benchmark MovieLens 1M dataset, containing 1,000,209 ratings from 6,040 users across 3,706 active movies. It is the gold-standard public dataset for collaborative filtering research."*

---

#### Cell 3: Interaction Matrix Sparsity & Stratified Train/Test Split

```python
num_users = ratings['UserID'].nunique()
num_movies = ratings['MovieID'].nunique()
num_ratings = len(ratings)
sparsity = (1.0 - (num_ratings / (num_users * num_movies))) * 100.0

print(f'Unique users: {num_users:,}')
print(f'Unique movies: {num_movies:,}')
print(f'Total ratings: {num_ratings:,}')
print(f'Matrix Sparsity: {sparsity:.2f}%')

train_ratings, test_ratings = train_test_split(
    ratings, test_size=0.20, random_state=RANDOM_SEED, stratify=ratings['UserID']
)
print(f'Training ratings: {len(train_ratings):,}')
print(f'Test ratings: {len(test_ratings):,}')

user_item_matrix = train_ratings.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)
print(f'User-Item matrix shape: {user_item_matrix.shape}')
```

- **Code line / operation**:
  1. Computes total catalog dimensions and matrix sparsity ($95.53\%$).
  2. Executes an 80/20 train/test split stratified on `UserID`.
  3. Pivots training ratings into a $6,040 \times 3,683$ user-item interaction matrix with missing entries filled with 0.
- **Why it is used**: Quantifying sparsity justifies dimensionality reduction. Stratification by `UserID` guarantees that every single user has interaction history in both train and test sets, preventing the unobserved cold-start problem during evaluation.
- **Formula**:
$$\text{Sparsity (\%)} = \left(1 - \frac{|\mathcal{R}|}{|\mathcal{U}| \times |\mathcal{I}|}\right) \times 100 = \left(1 - \frac{1,000,209}{6,040 \times 3,706}\right) \times 100 = 95.53\%$$
- **Output interpretation**: The matrix is $95.53\%$ sparse; 800,167 interactions are assigned to training, and 200,042 are reserved for evaluation.
- **Viva question**: *"Why did you fill unobserved ratings with 0? Does that mean the user hated the movie?"*
  - **Answer**: *"No. In collaborative filtering without explicit bias terms, 0 represents an unobserved interaction, not a 0-star rating. While imputing 0 shifts raw reconstructed ratings downward, the relative ordering of scores is preserved for top-K ranking."*

---

#### Cell 4: Latent Factor Decomposition & Score Reconstruction

```python
svd = TruncatedSVD(n_components=20, random_state=RANDOM_SEED)
user_factors = svd.fit_transform(user_item_matrix)
item_factors = svd.components_

predicted_ratings = svd.inverse_transform(user_factors)
predicted_df = pd.DataFrame(
    predicted_ratings,
    index=user_item_matrix.index,
    columns=user_item_matrix.columns
)

print(f'User factors shape: {user_factors.shape}')
print(f'Item factors shape: {item_factors.shape}')
print(f'Explained variance ratio sum: {svd.explained_variance_ratio_.sum():.4f}')
print(f'Predicted matrix shape: {predicted_df.shape}')
display(predicted_df.iloc[:5, :8].round(2))
```

- **Code line / operation**: Uses scikit-learn's `TruncatedSVD` with $k=20$ components to project the $6,040 \times 3,683$ matrix into 20 latent dimensions, then reconstructs predicted preference scores $\hat{\mathbf{R}} = \mathbf{P}\mathbf{Q}$.
- **Why it is used**: Matrix factorization maps both users and movies into a shared latent semantic space, extracting fundamental taste factors (e.g. action intensity, drama, vintage) and smoothing rating noise.
- **Formula**:
$$\mathbf{R}_{m \times n} \approx \mathbf{P}_{m \times k} \mathbf{Q}_{k \times n} = (\mathbf{U}_k \mathbf{\Sigma}_k) \mathbf{V}_k^T$$
where predicted score for user $u$ on movie $i$ is:
$$\hat{r}_{ui} = \mathbf{p}_u \cdot \mathbf{q}_i^T = \sum_{f=1}^{20} p_{uf} q_{if}$$
- **Output interpretation**: `user_factors` is $6,040 \times 20$, `item_factors` is $20 \times 3,683$, and `predicted_df` gives continuous preference predictions for all user-movie combinations.
- **Viva question**: *"Why did you select 20 components for Truncated SVD?"*
  - **Answer**: *"20 components provide the optimal balance between capturing dominant latent genre clusters (explaining ~16.1% of matrix variance) and preventing overfitting on sparse ratings, while computing in under two seconds."*

---

#### Cell 5: Recommendation Function Definition with Input Validation

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

- **Code line / operation**: Defines the core personalized recommendation function: validates that $n$ is a positive integer and that $user\_id$ exists, drops all movies already rated by the user in training, clamps $n$ to available unseen items, and returns the top $n$ movies sorted descending by predicted preference score.
- **Why it is used**: Recommending already-watched movies provides negative discovery utility. Input validation ensures robustness against invalid parameters or missing IDs without raising tracebacks.
- **Algorithm**:
  1. Validate $u \in \mathcal{U}$ and $n \in \mathbb{Z}^+$.
  2. $\mathcal{I}_{\text{seen}}(u) = \{i \mid (u, i) \in \mathcal{R}_{\text{train}}\}$
  3. $\mathcal{I}_{\text{unseen}}(u) = \mathcal{I} \setminus \mathcal{I}_{\text{seen}}(u)$
  4. Rank $i \in \mathcal{I}_{\text{unseen}}(u)$ by $\hat{r}_{ui}$ descending and return top $\min(n, |\mathcal{I}_{\text{unseen}}|)$.
- **Viva question**: *"Why do you explicitly exclude already-rated movies?"*
  - **Answer**: *"The objective of a discovery recommendation feed is to introduce users to unobserved content. Re-recommending movies the user already rated provides zero incremental utility."*

---

#### Cell 6: Personalized SVD Recommendation Demonstration

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

- **Code line / operation**: Executes the personalized collaborative filtering demo for User 42 and tests error handling with invalid user `99999` and non-positive `n=0`.
- **Output interpretation**: User 42 receives personalized recommendations dominated by iconic Sci-Fi and Action titles (*Star Trek: The Wrath of Khan*, *The Matrix*, *Star Trek IV*, *Stargate*, *Dune*) with predicted preference scores between 4.09 and 4.38; invalid inputs print descriptive error messages without crashing.
- **Viva question**: *"How does the system personalize recommendations for User 42?"*
  - **Answer**: *"By taking the dot product between User 42's 20-dimensional latent taste vector and each movie's 20-dimensional factor vector, filtering out the movies User 42 already rated, and ranking the remainder."*

---

#### Cell 7: Content-Based Movie Similarity Demonstration with Validation

```python
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

- **Code line / operation**: Transforms movie genres into multi-hot binary vectors, computes pairwise cosine similarity, and builds a robust title-matching function supporting exact matching, case insensitivity, leading article stripping (*The Matrix* -> *Matrix, The*), token matching (*Toy Story* -> *Toy Story (1995)*), candidate exclusion, and boundary error handling.
- **Why we use it**: Demonstrates **content-based movie similarity** ("More Like This" carousel), which is fundamentally different from collaborative user personalization.
- **Formula**:
$$\text{Cosine Similarity}(\mathbf{g}_A, \mathbf{g}_B) = \frac{\mathbf{g}_A \cdot \mathbf{g}_B}{\|\mathbf{g}_A\| \|\mathbf{g}_B\|} = \frac{\sum_{k=1}^D g_{Ak} g_{Bk}}{\sqrt{\sum_{k=1}^D g_{Ak}^2} \sqrt{\sum_{k=1}^D g_{Bk}^2}}$$
where $\mathbf{g}_i \in \{0, 1\}^D$ is the binary multi-hot genre vector for movie $i$.
- **Why the input movie itself is excluded**: The similarity of any item with itself is identically $1.0$. Returning the query item itself is trivial and unhelpful.
- **Viva question**: *"How does Movie-to-Movie Content Similarity differ from SVD Collaborative Filtering?"*
  - **Answer**: *"Movie similarity requires zero user ratings; it relies strictly on item metadata (multi-hot genres) via cosine similarity. SVD collaborative filtering requires collective interaction history across all users to infer latent taste factors."*

---

#### Cell 8: Baseline Accuracy & Retrieval Metrics (RMSE, Precision@K, Recall@K)

```python
valid_test = test_ratings[test_ratings['MovieID'].isin(predicted_df.columns)]
user_idx_map = {uid: i for i, uid in enumerate(predicted_df.index)}
movie_idx_map = {mid: i for i, mid in enumerate(predicted_df.columns)}

u_indices = [user_idx_map[uid] for uid in valid_test['UserID'].values]
m_indices = [movie_idx_map[mid] for mid in valid_test['MovieID'].values]

y_true = valid_test['Rating'].values
y_pred = predicted_df.values[u_indices, m_indices]
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

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

baseline_summary_df = pd.DataFrame({
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

print('Baseline Recommendation Performance')
display(baseline_summary_df)
```

- **Code line / operation**: Evaluates rating prediction error via vectorized RMSE, computes top-K retrieval metrics (defining relevant items as test ratings $\ge 4.0$), and displays the baseline summary table.
- **Formulas**:
$$\text{RMSE} = \sqrt{\frac{1}{|\mathcal{T}|} \sum_{(u, i) \in \mathcal{T}} (r_{ui} - \hat{r}_{ui})^2}$$
$$\text{Precision@}K = \frac{1}{|\mathcal{U}_{\text{eval}}|} \sum_{u \in \mathcal{U}_{\text{eval}}} \frac{|\text{Top-}K(u) \cap \mathcal{R}_{\text{test, rel}}(u)|}{K}$$
$$\text{Recall@}K = \frac{1}{|\mathcal{U}_{\text{eval}}|} \sum_{u \in \mathcal{U}_{\text{eval}}} \frac{|\text{Top-}K(u) \cap \mathcal{R}_{\text{test, rel}}(u)|}{|\mathcal{R}_{\text{test, rel}}(u)|}$$
- **Output interpretation**:
  - **RMSE**: **2.7013**
  - **Precision@5**: **0.3441** ($34.41\%$)
  - **Recall@5**: **0.1277** ($12.77\%$)
  - **Precision@10**: **0.2844** ($28.44\%$)
  - **Recall@10**: **0.1971** ($19.71\%$)
- **Viva question**: *"Why are Precision@K and Recall@K more critical than RMSE for a platform like Netflix?"*
  - **Answer**: *"Streaming users interact with a top-10 visual tray, not numerical rating predictions. Precision@K and Recall@K directly measure the accuracy of the items actually shown to the user."*

---

### 3. Part 1 Evaluation Metrics Master Reference

| Metric | Mathematical Formula | Definition | Direction | Value in Notebook | Viva Voce Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RMSE** | $\sqrt{\frac{1}{N}\sum (r - \hat{r})^2}$ | Rating prediction error magnitude | **Lower is better** | **2.7013** | *"Measures star rating error on held-out test data. Zero imputation shifts predictions toward zero, which raises RMSE but preserves ranking order."* |
| **Precision@5** | $\frac{|\text{Hits}@5|}{5}$ | Fraction of top 5 recommendations that are relevant ($\ge 4$) | **Higher is better** | **0.3441** | *"Over 34% of the movies in the top 5 list are confirmed user favorites in the test set."* |
| **Recall@5** | $\frac{|\text{Hits}@5|}{|\text{Relevant Test}|}$ | Proportion of all user favorites retrieved in top 5 | **Higher is better** | **0.1277** | *"Top 5 captures 12.8% of all movies the user rated highly in the test set."* |
| **Precision@10** | $\frac{|\text{Hits}@10|}{10}$ | Fraction of top 10 recommendations that are relevant ($\ge 4$) | **Higher is better** | **0.2844** | *"28.4% of top 10 recommendations are confirmed favorites."* |
| **Recall@10** | $\frac{|\text{Hits}@10|}{|\text{Relevant Test}|}$ | Proportion of all user favorites retrieved in top 10 | **Higher is better** | **0.1971** | *"Expanding the list from 5 to 10 increases captured favorites from 12.8% to 19.7%."* |

---

### 4. Part 1 Viva Voce Anticipated Questions & Answers

#### Q1: Why did you use Truncated SVD instead of Standard SVD?
> **Answer**: *"Standard SVD requires computing all singular values and full dense matrices, which is computationally prohibitive on a large matrix. Truncated SVD computes only the top $k=20$ dominant singular vectors directly via randomized algorithms, keeping execution to under two seconds."*

#### Q2: What happens if an unknown movie title is entered into `get_similar_movies()`?
> **Answer**: *"The function validates the input, checks exact matches, checks case-insensitive substrings, attempts leading article stripping, and attempts token matching. If no match is found, it prints `'Movie not found. Please enter a valid MovieLens movie title.'` and returns `None` without crashing."*

#### Q3: What happens if $n \le 0$ is passed to `recommend_movies()`?
> **Answer**: *"The function checks `if not isinstance(n, int) or n <= 0:`, prints `'n must be a positive integer.'`, and returns `None`, guaranteeing clean boundary handling."*

---

# PART 2 — CAUSAL AI + UPLIFT MODELING

---

### 5. Part 2 Conceptual Framing: From Preference to Causality

#### The Flaw of Conventional Observational Recommenders
A conventional recommendation system (Part 1 SVD) optimizes for **observational engagement**:
$$\text{Objective: } \max \hat{r}_{ui}$$
This causes the algorithm to recommend movies that users **already planned to watch anyway** (such as ubiquitous blockbusters). While the observational precision of such recommendations looks high, their **incremental business value is zero**. 

#### The Causal Uplift Paradigm
Part 2 introduces **Uplift Modeling (Incremental Treatment Effect Estimation)**:
$$\text{Objective: } \max \tau(X_{ui}) = \mu_1(X_{ui}) - \mu_0(X_{ui})$$
where:
- $\mu_1(X_{ui}) = P(Y_{ui} = 1 \mid X_{ui}, W_{ui} = 1)$ is the probability of positive engagement **when recommended** (Treatment).
- $\mu_0(X_{ui}) = P(Y_{ui} = 1 \mid X_{ui}, W_{ui} = 0)$ is the probability of positive engagement **without recommendation** (Organic Control).
- $\tau(X_{ui})$ is the **Individual Treatment Effect (ITE) / Uplift**, representing the net increase in engagement caused *directly* by the recommendation.

#### Fair Comparison Protocol
To guarantee a scientifically sound, fair comparison:
1. **Shared Candidate Universe**: Both SVD Baseline and Proposed Uplift policies compete over the **exact same pool of top-25 unseen candidates** extracted from Part 1 SVD.
2. **Identical List Length**: Both policies recommend exactly $K=10$ movies per user.
3. **Disjoint User Cohorts**: An 80/20 user-level randomized split ensures that evaluation users are never seen during causal model training.
4. **Zero Data Leakage**: Models generate recommendations using only predicted preference and estimated uplift scores. True potential outcomes are joined **post-hoc strictly during evaluation**.

---

### 6. Part 2 Section-by-Section Code & Theory Mapping

---

#### Cell 9: Randomized User Split & Candidate Generation from SVD Pool

```python
candidate_pool_size = 25
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

print(f'Causal training candidate pairs: {len(train_feat_df):,}')
print(f'Causal evaluation candidate pairs: {len(eval_feat_df):,}')
display(train_feat_df[feature_cols].head())
```

- **Code line / operation**:
  1. Performs a reproducible 80/20 randomized user-level split (1,600 training users, 400 evaluation users) using `train_test_split`.
  2. Confirms mathematically that user sets are disjoint (`len(set(train) & set(test)) == 0`).
  3. Extracts top 25 unseen candidate movies per user from SVD scores (40,000 training pairs, 10,000 evaluation pairs).
  4. Attaches 8 pre-treatment user, movie, and recommender features strictly from historical training interactions and metadata.
- **Why it is used**: Splitting at the user level prevents user interaction leakage across train and evaluation folds. Candidate extraction guarantees both policies compete over the identical candidate set.
- **Viva question**: *"Why did you split at the user level rather than splitting interaction rows?"*
  - **Answer**: *"If we split interaction rows randomly, candidate pairs for the same user would appear in both training and evaluation sets, causing user-level data leakage. Splitting at the user level guarantees that the 400 evaluation users are completely held out from model training."*

---

#### Cell 10: Randomized Treatment Assignment & Potential Outcomes Simulation

```python
train_feat_df['W'] = np.random.binomial(1, 0.5, size=len(train_feat_df))

scaler_dgp = StandardScaler()
X_tr_std = scaler_dgp.fit_transform(train_feat_df[feature_cols])

z0_tr = -0.8 + 0.35 * X_tr_std[:, 2] + 0.25 * X_tr_std[:, 6] + 0.15 * X_tr_std[:, 3]
p0_tr = 1 / (1 + np.exp(-z0_tr))
z1_tr = -0.4 + 0.10 * X_tr_std[:, 2] + 0.50 * X_tr_std[:, 6] + 0.30 * X_tr_std[:, 3] - 0.20 * X_tr_std[:, 7]
p1_tr = 1 / (1 + np.exp(-z1_tr))

prob_train = np.where(train_feat_df['W'] == 1, p1_tr, p0_tr)
train_feat_df['Y'] = np.random.binomial(1, prob_train)

print('Treatment assignment distribution:')
print(train_feat_df['W'].value_counts(normalize=True).round(3))
print(f'Overall positive engagement rate: {train_feat_df["Y"].mean():.4f}')
print(f'Treatment group positive rate: {train_feat_df[train_feat_df["W"] == 1]["Y"].mean():.4f}')
print(f'Control group positive rate: {train_feat_df[train_feat_df["W"] == 0]["Y"].mean():.4f}')
```

- **Code line / operation**: Simulates 50/50 randomized treatment assignment ($W \in \{0, 1\}$) and models potential outcome probabilities:
  - Organic control response $p_0(X) = \sigma(z_0(X))$: Driven by high movie popularity (`movie_train_count`, index 2) and baseline preference (`svd_score`, index 6).
  - Treatment response $p_1(X) = \sigma(z_1(X))$: Incorporates recommendation discovery lift, which provides the highest incremental boost on high-preference items that are not already saturated by massive popularity.
  - Realized binary engagement outcome $Y \sim \text{Bernoulli}(p)$: Realizes $Y = W \cdot Y(1) + (1-W) \cdot Y(0)$.
- **Theory**: Neyman-Rubin Potential Outcomes Framework. Under 50/50 randomized treatment, treatment assignment is independent of potential outcomes ($W \perp (Y(1), Y(0)) \mid X$), satisfying unconfoundedness.
- **Viva question**: *"How do you simulate potential outcomes?"*
  - **Answer**: *"We model organic control response p0 as driven by popularity and baseline preference, while treatment response p1 adds recommendation discovery lift. The realized outcome Y is simulated according to treatment assignment W."*

---

#### Cell 11: Two-Model Causal Estimator (T-Learner) & Individual Uplift Estimation

```python
model_control = make_pipeline(StandardScaler(), LogisticRegression(random_state=RANDOM_SEED, max_iter=500))
model_treatment = make_pipeline(StandardScaler(), LogisticRegression(random_state=RANDOM_SEED, max_iter=500))

ctrl_data = train_feat_df[train_feat_df['W'] == 0]
treat_data = train_feat_df[train_feat_df['W'] == 1]

model_control.fit(ctrl_data[feature_cols], ctrl_data['Y'])
model_treatment.fit(treat_data[feature_cols], treat_data['Y'])

eval_feat_df['mu0'] = model_control.predict_proba(eval_feat_df[feature_cols])[:, 1]
eval_feat_df['mu1'] = model_treatment.predict_proba(eval_feat_df[feature_cols])[:, 1]
eval_feat_df['uplift'] = eval_feat_df['mu1'] - eval_feat_df['mu0']

print('Models trained successfully with standardized features.')
print(f'Model 0 (Control) intercept: {model_control.named_steps["logisticregression"].intercept_[0]:.4f}')
print(f'Model 1 (Treatment) intercept: {model_treatment.named_steps["logisticregression"].intercept_[0]:.4f}')
print()
print('Estimated Uplift Statistics across Evaluation Candidates:')
display(eval_feat_df[['mu0', 'mu1', 'uplift']].describe().round(4))
```

- **Code line / operation**:
  1. Instantiates two scikit-learn pipelines combining `StandardScaler` with `LogisticRegression`.
  2. Fits Model 0 strictly on control candidate pairs ($W=0$) and Model 1 strictly on treatment candidate pairs ($W=1$).
  3. Predicts counterfactual probabilities $\hat{\mu}_0(X)$ and $\hat{\mu}_1(X)$ on the 10,000 evaluation candidate pairs.
  4. Calculates estimated uplift $\hat{\tau}(X) = \hat{\mu}_1(X) - \hat{\mu}_0(X)$.
- **Why we standardize**: Features like `movie_train_count` reach into the thousands, while `genre_count` ranges from 1 to 6. Without standardization, the L2 regularization penalty in LogisticRegression disproportionately shrinks coefficients of smaller-scale features.
- **Formulas**:
$$\hat{\mu}_0(X) = P(Y=1 \mid X, W=0) = \sigma(\mathbf{w}_0^T \tilde{\mathbf{X}} + b_0)$$
$$\hat{\mu}_1(X) = P(Y=1 \mid X, W=1) = \sigma(\mathbf{w}_1^T \tilde{\mathbf{X}} + b_1)$$
$$\hat{\tau}(X_{ui}) = \hat{\mu}_1(X_{ui}) - \hat{\mu}_0(X_{ui})$$
- **Viva question**: *"Why did you use a T-Learner instead of a single model with a treatment indicator?"*
  - **Answer**: *"A single model (S-Learner) with regularization often penalizes the treatment effect toward zero. The T-Learner fits separate models on control and treatment data, allowing feature relationships to differ completely between organic viewing and recommendation exposure."*

---

#### Cell 12: Interactive Causal-Uplift Demonstration & Compact Comparison (User 1861)

```python
demo_uid = 1861
user_demo = eval_feat_df[eval_feat_df['UserID'] == demo_uid].merge(movies[['MovieID', 'Title', 'Genres']], on='MovieID')

svd_top5 = user_demo.sort_values('svd_score', ascending=False).head(5).copy()
svd_top5['Rank'] = range(1, 6)

upl_top5 = user_demo.sort_values('uplift', ascending=False).head(5).copy()
upl_top5['Rank'] = range(1, 6)

print(f'PART 2 - OUR CAUSAL-UPLIFT DEMO (User {demo_uid})')
print()
print('TRADITIONAL SVD')
display(svd_top5[['Rank', 'Title', 'svd_score']].rename(columns={'Title': 'Movie', 'svd_score': 'SVD Score'}).round(3))
print()
print('OUR CAUSAL-UPLIFT METHOD')
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

print()
print(f'COMPACT DEMONSTRATION COMPARISON TABLE (User {demo_uid})')
display(comp_table)
```

- **Code line / operation**: Evaluates Demo User 1861 from the held-out evaluation set over the identical 25-candidate pool, producing SVD top 5, Uplift top 5, and the compact comparison table.
- **The Core Viva Result**:

| Movie | SVD Preference | Estimated Uplift | Selected by SVD | Selected by Uplift | Viva Demonstration Story |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **American Graffiti (1973)** | 3.175 | **0.2932** | **Yes** | **Yes** | High preference AND highest uplift |
| **One Flew Over the Cuckoo's Nest (1975)** | 3.444 | **0.2704** | **Yes** | **Yes** | Strong preference and strong incrementality |
| **Bridge on the River Kwai, The (1957)** | 2.778 | **0.2682** | No | **Yes** | Tailored discovery gem promoted by uplift |
| **2001: A Space Odyssey (1968)** | 3.287 | **0.2630** | **Yes** | **Yes** | Balanced high performer |
| **Dances with Wolves (1990)** | 3.068 | **0.2433** | No | **Yes** | Promoted by uplift due to high discovery lift |
| **Back to the Future (1985)** | **3.474** | 0.1863 | **Yes** | No | Top SVD pick dropped due to organic saturation |
| **Saving Private Ryan (1998)** | 3.229 | 0.1803 | **Yes** | No | High baseline viewership dropped by uplift |

- **Viva answer**: *"This table demonstrates the central thesis of our project: 'Back to the Future' has the highest SVD preference score (3.474), but because it is an iconic blockbuster that the user would watch organically anyway, its estimated uplift is lower (0.1863). The uplift model de-selects it and replaces it with 'The Bridge on the River Kwai' (+0.2682), which produces higher incremental response."*

---

#### Cell 13: Visual Demonstration: SVD Preference vs. Estimated Uplift

```python
fig, ax1 = plt.subplots(figsize=(10, 5))
y_pos = np.arange(len(comp_table))
bar_height = 0.35

ax1.barh(y_pos - bar_height/2, comp_table['SVD Preference'], bar_height, label='SVD Preference Score', color='#4A90E2', edgecolor='black')
ax1.set_xlabel('SVD Preference Score', color='#1B4F72', fontsize=11, fontweight='bold')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(comp_table['Movie'], fontsize=10)
ax1.invert_yaxis()

ax2 = ax1.twiny()
ax2.barh(y_pos + bar_height/2, comp_table['Estimated Uplift'], bar_height, label='Estimated Incremental Uplift', color='#7ED321', edgecolor='black')
ax2.set_xlabel('Estimated Incremental Uplift (CATE)', color='#1E8449', fontsize=11, fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

plt.title(f'Conventional Preference vs. Causal Uplift (Demo User {demo_uid})', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()
```

- **Code line / operation**: Renders a dual-axis horizontal grouped bar chart for Demo User 1861 comparing SVD Preference Score vs. Estimated Incremental Uplift.
- **Viva answer**: *"This chart visually contrasts 'What the user likes' against 'What recommendation causes behavior change,' making the divergence between static preference and causal lift obvious."*

---

#### Cell 14: Counterfactual Policy Evaluation with True Simulated Uplift

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

print(f'Evaluation complete across {len(test_causal_users)} held-out users.')
print(f'Mean top-10 policy overlap: {np.mean(overlap_list):.2f} / 10 movies')
```

- **Code line / operation**:
  1. Computes simulator true potential outcomes ($p_0, p_1, \tau_{\text{true}} = p_1 - p_0$) on evaluation candidates.
  2. Generates top-10 recommendations for SVD (using `svd_score`) and Proposed Uplift (using estimated `uplift`) **independently without access to true potential outcomes**.
  3. Joins the true simulated values strictly post-hoc to evaluate:
     - `Mean Predicted Treatment Response`: $\frac{1}{K} \sum \hat{\mu}_1(X)$
     - `Mean Estimated Uplift@10`: $\frac{1}{K} \sum \hat{\tau}(X)$
     - `Mean True Simulated Uplift@10`: $\frac{1}{K} \sum \tau_{\text{true}}(X)$
- **Why post-hoc joining is mandatory**: Guarantees zero data leakage; true simulated values evaluate the policy without influencing recommendation rankings.
- **Viva answer**: *"We generate recommendations using only predicted SVD preference and estimated model uplift. We then join true simulated uplift post-hoc to independently verify whether the proposed policy actually achieved higher incremental impact."*

---

#### Cell 15: Comparative Policy Performance Scorecard

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
| **Precision@10** | **0.2775** | 0.2725 | Negligible trade-off (-0.5%) to explore catalog items |
| **Recall@10** | **0.1966** | 0.1933 | Retains 98.3% of baseline retrieval coverage |
| **Mean Predicted Treatment Response** | **0.5220** | 0.5042 | Gross probability of positive engagement under exposure |
| **Mean Estimated Uplift@10** | 0.1467 | **0.1538** | **+4.8% relative gain** in model-estimated uplift |
| **Mean True Simulated Uplift@10** | 0.1480 | **0.1576** | **+6.5% relative gain** in true incremental lift |
| **Catalog Coverage** | 10.20% | **11.12%** | **+9.0% broader catalog exploration** |
| **Ranking Objective** | Predicted preference | Predicted incremental response | Observational correlation vs. Causal intervention |

- **Viva answer**: *"Our primary metric, Mean True Simulated Uplift@10, increases from 0.1480 under SVD to 0.1576 under Proposed Uplift. This confirms that our uplift layer successfully selects movies with greater incremental causal impact while maintaining 27.25% Precision@10."*

---

#### Cell 16: Primary Visualizations: True Incremental Lift@10 & Diagnostics

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sample_eval = eval_feat_df.sample(n=min(2000, len(eval_feat_df)), random_state=RANDOM_SEED)
axes[0, 0].scatter(sample_eval['svd_score'], sample_eval['uplift'], alpha=0.35, color='#4A90E2', edgecolors='none')
axes[0, 0].set_title('SVD Preference Score vs Estimated Uplift', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('SVD Preference Score', fontsize=10)
axes[0, 0].set_ylabel('Estimated Uplift', fontsize=10)
axes[0, 0].grid(True, linestyle='--', alpha=0.5)

policies = ['SVD Baseline', 'Proposed Uplift']
true_uplift_vals = [np.mean(svd_true_upl_list), np.mean(upl_true_upl_list)]
colors = ['#F5A623', '#7ED321']
bars = axes[0, 1].bar(policies, true_uplift_vals, color=colors, edgecolor='black', width=0.5)
axes[0, 1].set_title('True Incremental Lift@10: SVD vs Proposed Uplift', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Recommendation Strategy', fontsize=10)
axes[0, 1].set_ylabel('Mean True Uplift', fontsize=10)
for bar in bars:
    yval = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.003, f'{yval:.4f}', ha='center', fontsize=10, fontweight='bold')

metrics = ['Precision@10', 'Recall@10']
svd_pr = [np.mean(svd_p10_list), np.mean(svd_r10_list)]
upl_pr = [np.mean(upl_p10_list), np.mean(upl_r10_list)]
x = np.arange(len(metrics))
width = 0.35
axes[1, 0].bar(x - width/2, svd_pr, width, label='SVD Baseline', color='#4A90E2', edgecolor='black')
axes[1, 0].bar(x + width/2, upl_pr, width, label='Proposed Uplift', color='#BD10E0', edgecolor='black')
axes[1, 0].set_title('Precision@10 & Recall@10 Comparison', fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(metrics)
axes[1, 0].set_ylabel('Score', fontsize=10)
axes[1, 0].legend()
for i, v in enumerate(svd_pr):
    axes[1, 0].text(i - width/2, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)
for i, v in enumerate(upl_pr):
    axes[1, 0].text(i + width/2, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)

axes[1, 1].hist(overlap_list, bins=np.arange(-0.5, 11.5, 1), color='#50E3C2', edgecolor='black', alpha=0.85)
axes[1, 1].set_title('Top-10 Recommendation Overlap Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Number of Shared Movies in Top-10', fontsize=10)
axes[1, 1].set_ylabel('User Count', fontsize=10)
axes[1, 1].set_xticks(range(0, 11))
axes[1, 1].axvline(np.mean(overlap_list), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(overlap_list):.2f}')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
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

#### Q6: What are the main limitations of this causal recommendation study?
> **Answer**: *"First, MovieLens is an observational ratings dataset, so treatment exposure and counterfactual outcomes are simulated rather than logged from a live production A/B test. Second, user fatigue and position bias in UI carousels are not modeled. However, the conceptual architecture demonstrates exactly how production uplift layers can be integrated with collaborative filtering."*
