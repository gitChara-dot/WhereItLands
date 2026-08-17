# WhereItLands

An end-to-end Machine Learning and statistical inference system designed to model, simulate, and predict international football match outcomes. The platform combines tournament-weighted dynamic Elo ratings, rolling historical feature engineering, multi-model stacking regression, and a bivariate Poisson distribution corrected with the Coles-Dixon model for low-scoring match dependency.

**Live Demo:** [https://whereitlands.onrender.com](https://whereitlands.onrender.com)  
**Interactive API Documentation:** [https://whereitlands.onrender.com/docs](https://whereitlands.onrender.com/docs)

---

## 1. System Overview and Architecture

WhereItLands is structured as a decoupled, production-oriented pipeline comprising three primary tiers: offline data preparation and feature extraction, supervised ensemble training, and a sub-10ms inference microservice serving both a REST API and a native web interface.

```
+-----------------------------------------------------------------------------+
|                                OFFLINE TIER                                 |
|                                                                             |
|  [ Raw Results CSV ]                                                        |
|           |                                                                 |
|           v                                                                 |
|  [ Data Pipeline ] ---> [ Dynamic Elo Engine (Tournament K-Weights) ]       |
|           |                                                                 |
|           v                                                                 |
|  [ Feature Engineering: 5-Match Rolling Averages, Defense Weights, Streaks] |
|           |                                                                 |
|           v                                                                 |
|  [ Artifact Vault: team_history.parquet, elo_system.joblib ]                |
|           |                                                                 |
|           v                                                                 |
|  [ Stacking Ensemble Training: XGBoost + LightGBM + ExtraTrees -> RidgeCV ] |
|           |                                                                 |
|           v                                                                 |
|  [ Model Artifacts: home_stack.joblib, away_stack.joblib ]                  |
+-----------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------+
|                                INFERENCE TIER                               |
|                                                                             |
|  [ FastAPI Server (Startup Lifespan: Load All Artifacts to RAM) ]           |
|           |                                                                 |
|           +---> [ Endpoint: POST /prediction ]                              |
|           |           |                                                     |
|           |           v                                                     |
|           |     [ Point-in-Time Rolling Feature Extraction ]                |
|           |           |                                                     |
|           |           v                                                     |
|           |     [ Stacking Inference: Lambda_Home, Lambda_Away ]            |
|           |           |                                                     |
|           |           v                                                     |
|           |     [ Bivariate Poisson Matrix + Coles-Dixon Correction ]       |
|           |           |                                                     |
|           |           v                                                     |
|           |     [ Output: 3-Way Chances (1X2) + Top-N Exact Scorelines ]    |
|           |                                                                 |
|           +---> [ Native Frontend: HTML5 / CSS3 / Vanilla JS Fetch Client ] |
+-----------------------------------------------------------------------------+
```

---

## 2. Statistical and Machine Learning Methodology

### 2.1 Dynamic Elo Engine
Team strength is updated chronologically through a customized logistic Elo framework that applies dynamic tournament importance multipliers ($K$) and margin-of-victory scaling.

The expected outcome $E_A$ for team $A$ playing team $B$ with optional pitch advantage $\gamma$ is computed as:

$$E_A = \frac{1}{1 + 10^{-(R_A + \gamma - R_B) / 400}}$$

Match weight $K$ is assigned based on competitive context:
* FIFA World Cup Finals: $K = 60$
* Continental Championships (e.g., Copa America, UEFA Euro): $K = 50$
* Major Qualifiers and Nations Leagues: $K = 40$
* Regional Tournaments: $K = 25$
* Minor Qualifiers: $K = 15$
* Tier-1 Friendlies and Invitationals: $K = 10$
* Exhibition and Unaffiliated Matches: $K = 5$

The goal difference multiplier $M(d)$ accounts for performance margins:

$$
M(d) = 
\begin{cases} 
1.0 & \text{if } d \le 1 \\
1.5 & \text{if } d = 2 \\
1.75 & \text{if } d = 3 \\
1.75 + \dfrac{d - 3}{8} & \text{if } d > 3 
\end{cases}
$$

### 2.2 Expected Goal Rate Estimation ($\lambda$)
To predict continuous goal-scoring intensity parameters ($\lambda_{\text{home}}, \lambda_{\text{away}}$), a two-level `StackingRegressor` architecture is employed for each side.

* **Base Estimators:**
  * **XGBoost Regressor:** Configured with `count:poisson` objective and `poisson-nloglik` loss function.
  * **LightGBM Regressor:** Configured with Poisson objective and dynamic leaf tuning.
  * **ExtraTrees Regressor:** Wrapped in a `TransformedTargetRegressor` using a $\log(1 + y)$ transformation to stabilize variance across high-scoring tail distributions.
* **Meta-Estimator:**
  * **RidgeCV:** Cross-validated L2-regularized linear blender over candidate alphas ($0.01$ to $500.0$) using expanding window `TimeSeriesSplit` cross-validation to eliminate lookahead bias.

### 2.3 Bivariate Poisson and Coles-Dixon Adjustment
Independent Poisson assumptions underestimate the probability of low-scoring outcomes (such as $0-0$ or $1-1$). The model builds a $6 \times 6$ bivariate probability matrix $P(X=x, Y=y)$ adjusted by the Coles-Dixon factor $\tau(x, y; \rho)$:

$$P(X=x, Y=y) = \frac{\lambda_h^x e^{-\lambda_h}}{x!} \cdot \frac{\lambda_a^y e^{-\lambda_a}}{y!} \cdot \tau(x, y; \rho)$$

Where the parameter $\rho = -0.05$ adjusts the joint probability of low-scoring cells:

$$
\tau(x, y; \rho) = 
\begin{cases} 
1 - \lambda_h \lambda_a \rho & \text{if } x = 0, y = 0 \\
1 + \lambda_h \rho & \text{if } x = 0, y = 1 \\
1 + \lambda_a \rho & \text{if } x = 1, y = 0 \\
1 - \rho & \text{if } x = 1, y = 1 \\
1.0 & \text{otherwise} 
\end{cases}
$$

Aggregate win, draw, and loss probabilities are calculated by summing the corresponding partitions of the matrix, while exact scoreline ranks are derived by sorting matrix coordinates in descending order of density.

---

## 3. Validation and Performance Benchmarks

The model was evaluated using a strict chronological split to measure true out-of-sample generalization. All matches prior to 2025-01-01 served as the training corpus, while matches from 2025 onward were held out as an unobserved test set.

| Metric / Evaluation Parameter | Result | Description |
| :--- | :--- | :--- |
| **Out-of-Sample 3-Way Accuracy (1X2)** | **64.24%** | Accuracy on test holdout set (post-2025 competitive fixtures). |
| **Cross-Validation Strategy** | **TimeSeriesSplit (5 Splits)** | Expanding-window temporal CV to prevent data leakage during hyperparameter search. |
| **Median Inference Latency** | **< 8 ms** | Average execution time per prediction request under FastAPI lifespan caching. |
| **Artifact Memory Footprint** | **~76 MB** | In-memory RAM consumption for models, Elo state, and columnar Parquet history. |

---

## 4. API Reference

The backend exposes typed endpoints documented with OpenAPI (Swagger) specifications.

### `GET /health`
Returns the readiness state of the machine learning artifacts loaded in memory.

**Response:**
```json
{
  "status": "ready"
}
```

### `GET /teams`
Returns the complete list of unique national teams available in the preprocessed feature store.

**Response:**
```json
{
  "teams": [
    "Argentina",
    "Brazil",
    "France",
    "Germany",
    "Spain"
  ]
}
```

### `POST /prediction`
Executes single-match inference based on current ratings or historical point-in-time features.

**Request Payload:**
```json
{
  "home_team": "Argentina",
  "away_team": "France",
  "neutral": true,
  "iterations": 3,
  "date": "2024-07-14"
}
```

**Response Payload:**
```json
{
  "home_win_chance": 0.442,
  "draw_chance": 0.286,
  "away_win_chance": 0.272,
  "top_results": [
    [1, 1, 13.8],
    [1, 0, 12.4],
    [2, 1, 10.9]
  ]
}
```

---

## 5. Repository Structure

```
WhereItLands/
├── artifacts/                     # Serialized artifacts (models, Elo state, Parquet history)
│   ├── away_stack.joblib
│   ├── elo_system.joblib
│   ├── home_stack.joblib
│   └── team_history_dataframe.parquet
├── config/
│   └── config.yaml                # Tournament weights, hyperparameters, constants
├── data/
│   └── unprocessed/               # Historical match datasets
├── frontend/                      # Native web dashboard
│   ├── app.js                     # Asynchronous DOM controller and API fetch client
│   ├── index.html                 # Semantic UI structure
│   └── styles.css                 # Responsive design system
├── src/
│   ├── config.py                  # YAML loader utility
│   ├── data_processing/
│   │   └── pipeline.py            # Elo application, rolling averages, feature builder
│   ├── inference/
│   │   ├── api.py                 # FastAPI application with static files mount
│   │   ├── results_calc.py        # Point-in-time statistics and Poisson inference
│   │   └── schemas.py             # Pydantic request and response schemas
│   ├── math_utils/
│   │   ├── EloSystem.py           # Object-oriented Elo calculation engine
│   │   └── stats_utils.py         # Coles-Dixon and Poisson probability matrix utilities
│   └── training/
│       └── train_pipeline.py      # StackingRegressor training and tuning pipeline
├── tests/                         # Automated test suite (Pytest)
│   ├── test_api.py                # FastAPI endpoint integration tests
│   ├── test_math_utils.py         # Elo and Coles-Dixon unit tests
│   └── test_pipeline.py           # Data processing and filtering unit tests
├── .dockerignore
├── .gitattributes
├── .gitignore
├── Dockerfile                     # Multi-stage containerization blueprint
├── main.py                        # Pipeline CLI entry point
├── README.md
└── requirements.txt               # Pinned Python dependencies
```

---

## 6. Local Setup and Execution

### Prerequisites
* Python 3.10, 3.11, or 3.12 (or Docker)
* Git

### Option A: Standard Python Virtual Environment

1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/gitChara-dot/WhereItLands.git
   cd WhereItLands
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Start the local server:
   ```bash
   uvicorn src.inference.api:app --reload --port 8000
   ```

3. Access the services:
   * **Web Application:** `http://127.0.0.1:8000/`
   * **API Documentation:** `http://127.0.0.1:8000/docs`

### Option B: Running with Docker

1. Build the container image:
   ```bash
   docker build -t whereitlands .
   ```

2. Run the containerized service:
   ```bash
   docker run -p 8000:8000 whereitlands
   ```

3. Access the web interface at `http://localhost:8000/`.

---

## 7. Automated Testing

The repository includes a comprehensive unit and integration test suite using `pytest` and FastAPI `TestClient`.

To execute the test suite:

```bash
# Run all tests
python -m pytest

# Run with verbose reporting
python -m pytest -v
```

**Test Coverage Summary:**
* **`tests/test_math_utils.py`:** Validates logistic Elo symmetry, rating updates, Coles-Dixon boundary conditions, and Poisson probability matrix integration ($\sum P \approx 1.0$).
* **`tests/test_pipeline.py`:** Validates match filtering rules, chronological sorting, and team history aggregation.
* **`tests/test_api.py`:** Tests `GET /health`, `GET /teams`, `POST /prediction` with valid matchups, point-in-time historical dates, and edge case validation (identical or unrecorded teams).

---

## 8. Executing the Training Pipeline

To recompute Elo ratings, rebuild historical feature matrices, and retrain the stacking ensembles:

```bash
# Full execution (data pipeline + model training)
python main.py --mode train

# Fast artifact generation (updates Elo ratings and Parquet without model refitting)
python main.py --mode train --skip-training
```

---

## 9. License and Acknowledgments

This project is licensed under the MIT License. Match records and historical data are sourced from international football archives under open research licenses.