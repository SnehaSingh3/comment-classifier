# Comment Category Prediction

A multi-class NLP classification project that predicts the category of online comments using a combination of text features, categorical attributes, and engagement signals.

---

## Problem Statement

Given a dataset of online comments with associated metadata (race, religion, gender, upvotes, downvotes, emoticons, etc.), the goal is to classify each comment into one of **4 categories (labels 0–3)**. The dataset is imbalanced — Label 0 dominates at ~57.6% while Label 3 is rare at ~2.7%.

---

## Dataset

- **Source:** Kaggle Competition — *Comment Category Prediction Challenge*
- **Files:** `train.csv`, `test.csv`, `Sample.csv`
- **Key columns:**
  - `comment` — raw text of the comment
  - `race`, `religion`, `gender` — categorical demographic metadata
  - `emoticon_1`, `emoticon_2`, `emoticon_3` — emoticon usage flags
  - `upvote`, `downvote` — engagement signals
  - `if_1`, `if_2`, `disability` — additional binary indicators
  - `label` — target class (0, 1, 2, 3)

---

## Approach

### 1. Exploratory Data Analysis
- Analyzed class distribution (found significant imbalance)
- Visualized comment length and upvote distribution across labels
- Checked for null values and handled them appropriately

### 2. Feature Engineering
- Text features from `comment` column
- Filled missing categorical values (`race`, `religion`, `gender`) with `"none"`
- Retained numerical engagement features as-is

### 3. Preprocessing Pipeline
Used `sklearn.ColumnTransformer` combining:
- **TF-IDF Vectorizer** on `comment` — 40,000 features, bigrams `(1,2)`, `sublinear_tf=True`, `min_df=2`, `max_df=0.9`, English stop words removed
- **OneHotEncoder** on categorical columns (`race`, `religion`, `gender`)
- **Passthrough** for numerical features

### 4. Model Comparison

| Model                | Macro F1 Score |
|---------------------|---------------|
| Logistic Regression  | 0.7694        |
| Linear SVM           | 0.6798        |
| SGD Classifier       | 0.6248        |
| Random Forest        | 0.6497        |
| **LightGBM**         | **0.8031**    |

All models used `class_weight="balanced"` to handle class imbalance.

### 5. Best Model — LightGBM

```python
LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=50,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42
)
```

The final model was retrained on the full training set before generating test predictions.

---

## Key Results

- **Best Macro F1 Score: 0.8031** (LightGBM on validation set)
- LightGBM outperformed all other models, including Logistic Regression (0.7694), which was the second-best performer

---

## Tech Stack

- **Language:** Python 3
- **Platform:** Kaggle Notebooks
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `seaborn`, `matplotlib`

---

## Project Structure

```
├── comment-category-prediction_sneha.ipynb   # Main notebook
└── README.md
```

---

## How to Run

1. Clone the repository
2. Place `train.csv`, `test.csv`, and `Sample.csv` in the input directory
3. Run all cells in `comment-category-prediction_sneha.ipynb`
4. The final `submission.csv` will be saved in the working directory

---

## Author

Sneha Singh — IITM BS Data Science Program
