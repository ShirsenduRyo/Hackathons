# BigMart Sales Prediction — Approach Note
### Thought Process & Experimentation Steps
**Hackathon: Data Science — Learn & Compete | April 2026**

---

## 1. Problem Understanding

The objective is to predict `Item_Outlet_Sales` — the sales of 1,559 products across 10 BigMart outlets using 2013 transactional data. The dataset has 8,523 training rows and 5,681 test rows with no target provided for test. The metric is **RMSE on raw sales**.

Two immediate structural observations shaped the entire strategy:

- The sales distribution is heavily right-skewed (skewness ≈ 1.72), making **log1p transformation** of the target a necessity before modelling.
- Grocery Store outlets have a **3–4× lower mean sales** (≈340) compared to Supermarkets (≈2,316–3,694), representing fundamentally different sales dynamics that a single model would compromise on.

---

## 2. Exploratory Data Analysis

EDA was conducted in `01_EDA.ipynb` across 11 structured sections.

### 2.1 Data Quality Issues Identified

| Column | Issue | Fix |
|---|---|---|
| `Item_Weight` | 1,463 missing (~17%) | Per-item mean across combined train+test |
| `Item_Visibility` | 526 rows with exactly zero | Per-item mean of non-zero rows only |
| `Outlet_Size` | 2,410 missing (~28%) | Per-Outlet_Type mode + `Outlet_Size_Was_Missing` flag preserved |
| `Item_Fat_Content` | 5 inconsistent categories | Normalised to 2 canonical values; NC items → Non-Edible |

### 2.2 Key Signals from EDA

- **`Item_MRP`** is the strongest numeric predictor (Pearson r ≈ 0.57). Higher-priced items sell more.
- **`Outlet_Type` and `Outlet_Identifier`** drive massive sales variance — Supermarket Type3 outsells Grocery Stores by ~10×.
- **`Item_Visibility`** is right-skewed; log1p transformation normalises it.
- **Item_Identifier prefix** (FD / DR / NC) encodes food, drink, and non-consumable categories with distinct sales profiles.

---

## 3. Feature Engineering

All features were constructed to avoid target leakage. Key additions beyond raw columns:

| Feature | Rationale |
|---|---|
| `Outlet_Age` | 2013 − Establishment Year. Older outlets have established customer bases. |
| `Log_Item_Visibility` | log1p transform corrects right skew of visibility. |
| `Item_Visibility_MeanRatio` | Row visibility / item mean — captures relative shelf prominence. |
| `MRP_Per_Category` | MRP relative to item type average — pricing position signal. |
| `MRP_x_OutletType` | Interaction: pricing behaves differently across store types. |
| `MRP_Bin` (16 bins) | Quantile binning captures non-linear MRP effects. |
| `Item_Cat_x_Outlet` | Item category × outlet type interaction — e.g. non-food sells poorly in grocery. |
| `Is_Grocery` | Binary flag enabling hard split between retail environments. |
| `Outlet_Size_Was_Missing` | Binary flag: outlets that didn't report size cluster meaningfully. |
| `Outlet_Age_x_Type` | Outlet age × outlet type ordinal — older Type3 behaves differently from older Grocery. |
| `Weight_x_MRP` | Item weight × MRP — dense/expensive items (e.g. edible oil) have distinct sales patterns. |

### 3.1 KFold Target Encoding

Five high-cardinality columns — `Item_Identifier` (1,559 unique), `Outlet_Identifier` (10), `Item_Type` (16), `Item_Category`, and `Item_Cat_x_Outlet` — were mean-encoded using **5-fold out-of-fold encoding** with Laplace smoothing (α=10).

This is the **single highest-impact feature transformation**: it encodes historical sales signal directly into numeric features without any leakage, replacing naive label encoding which discards ordinal sales information entirely.

---

## 4. Modelling Strategy

### 4.1 Grocery Store Split

A hard split was introduced between Grocery Store rows (n=1,083, mean sales ≈340) and Supermarket rows (n=7,440, mean sales ≈2,500). Training separate models prevents the gradient from splitting its capacity between two vastly different sales scales, and allows each sub-model's hyperparameters to be tuned independently for its data distribution.

### 4.2 Hyperparameter Optimisation via Optuna

Each model was tuned using **Optuna** with a TPE (Tree-structured Parzen Estimator) sampler and `MedianPruner` to kill underperforming trials early. Six independent studies were run — one per model type per split — totalling **600 trials** (raised from 480; grocery budget increased to 100 trials to match supermarket, as small data warrants more regularisation search). The objective in every study was 5-fold out-of-fold RMSE on the log-transformed target.

Search space corrections applied in v8:
- XGB `gamma` range narrowed from 0–5 → 0–1 (values >1 prune nearly all splits on this dataset size)
- XGB `colsample_bytree` and `colsample_bylevel` floors raised from 0.4 → 0.5 to avoid simultaneous column starvation

### 4.3 Three Standalone Models

XGBoost, LightGBM, and CatBoost were each benchmarked independently before combining, to identify which learner dominates on each split and establish individual model ceilings.

### 4.4 Stacking Ensemble

The final architecture stacks all three base learners. Out-of-fold predictions from XGBoost, LightGBM, and CatBoost feed a **BayesianRidge meta-learner**. 

From v8, the meta-learner receives **5 inputs** instead of 3: the three OOF prediction columns plus `Item_MRP` and `TE_Outlet_Identifier`. This gives BayesianRidge real signal to differentiate predictions across price bands and outlet-level effects — when all three base OOF columns are highly correlated (>0.97), the additional raw features provide the variance the meta-learner needs to improve on a simple average. OOF pairwise correlations are printed at runtime as a diagnostic.

All 5 meta inputs are passed through a `StandardScaler` before BayesianRidge. The ensemble is trained and evaluated independently per split.

### 4.5 Pseudo-labelling

After round-1 predictions are generated, a pseudo-label round is run on confident test rows — those whose predictions deviate more than 0.5σ from the mean. These rows are appended to training data and the full stacking pipeline is retrained. The pseudo-label submission is only adopted if its OOF RMSE improves on round-1; otherwise the original predictions are kept.

---

## 5. Evaluation, Normalisation & Feature Pruning

### 5.1 Why No Pre-hoc Correlation Filtering

With ~22 features, pre-hoc correlation filtering was deliberately avoided. Tree models handle correlated features natively — they simply select the more informative one at each split. Removing correlated features before training risks discarding complementary signal (e.g. `Item_MRP` and `TE_Item_Identifier` are correlated but each carries unique variance). **Permutation importance post-training** is a safer, model-aware alternative.

### 5.2 Permutation Importance

After final model fitting, permutation importance (20 repeats, `sklearn.inspection`) was computed per split. Features where shuffling causes no RMSE degradation are flagged as drop candidates. A **consensus drop list** identifies features useless in both splits simultaneously — a tighter criterion than single-split analysis.

From v7 output: `Is_Grocery` and `Item_Fat_Content` were the only consensus-drop candidates. These are dropped in v8.

### 5.3 Normalisation Scope

`StandardScaler` is applied **exclusively to the 5 meta-learner inputs**. Tree base models receive raw features — scaling has zero effect on split-based learners. The log1p target transformation serves as implicit output normalisation, stabilising variance and pulling the right tail of the sales distribution closer to Gaussian.

### 5.4 Prediction Clipping

Final predictions are clipped at the minimum observed training sales value (~33). This is a free RMSE improvement on the lower tail with no model changes required.

---

## 6. Iteration Path & RMSE Scores

| Notebook | Key Change | RMSE |
|---|---|---|
| `02_Solution` | GBM + RF + Ridge stacking (baseline) | 1178.9507 |
| `03_XGB_Standalone` | XGB standalone benchmark | 1180.5147 |
| `04_LightGBM_Standalone` | LightGBM standalone benchmark | 1182.0863 |
| `05_CatBoost__Standalone` | CatBoost standalone benchmark | 1180.7270 |
| `06_Full_Solution` | Target encoding + Grocery split + 3-model stack | 1176.7038 |
| `07_Full_Solution_SS` | + StandardScaler on meta inputs | 1176.7031 |
| `08_Full_Solution_v8` | + 3 new features + richer meta (5 inputs) + clip + pseudo-label + Optuna fixes | 1174.4941 |
| `10_Full_Solution_v10` | E65ntity Embeddings + Optuna fixes | 1165.0286 |


The two **highest-impact single changes** across the entire pipeline were:

1. **KFold target encoding** of `Item_Identifier` and `Outlet_Identifier`, which directly injects historical sales signal into model-readable features.
2. **Grocery Store hard split**, which eliminates the model's need to reconcile a 10× sales scale difference within a single loss function.

---

*BigMart Sales Prediction — Approach Note | April 2026*
