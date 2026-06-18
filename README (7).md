# 🔍 Fake Profile Detection in Social Media

A machine learning system that classifies social media profiles as **Real** or **Fake** using a Random Forest classifier trained on behavioral and metadata features (followers, posts, bio length, username patterns, etc.).

---

## 📁 Project Structure

```
fake-profile-detection/
├── fake_profile_detection.py     # Standalone script: train, evaluate, plot dashboard
├── fake_profile_detection.ipynb  # Jupyter notebook walkthrough (EDA + training + demo)
├── app.py                        # Flask web app with live prediction API
├── templates/
│   └── index.html                # (required by app.py — see Setup Notes below)
└── analysis_dashboard.png        # Generated 11-panel evaluation dashboard
```

---

## 🎯 What It Does

Given a social media profile's metadata, the model predicts whether the account is **fake** (bot/spam) or **real**, along with a confidence score and the top features driving that decision.

**Input features:**

| Feature | Description |
|---|---|
| `profile_pic` | Has a profile photo (0/1) |
| `nums_in_username` | Count of digits in username |
| `fullname_words` | Number of words in full name |
| `fullname_nums` | Digits in full name |
| `bio_length` | Character length of bio |
| `external_url` | Has external link in bio (0/1) |
| `private` | Account is private (0/1) |
| `posts` | Total post count |
| `followers` | Follower count |
| `follows` | Following count |
| `account_age_days` | Days since account creation |
| `follower_follow_ratio` | Derived: followers / (follows + 1) |
| `posts_per_day` | Derived: posts / (account_age_days + 1) |

---

## 🧠 Model

**Algorithm:** Random Forest Classifier (scikit-learn)

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

**Dataset:** Synthetic, generated to mimic the structure of the [Kaggle Instagram Fake/Spammer/Genuine Accounts dataset](https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-accounts). 5,000 profiles, 50/50 class balance. Fake profiles are simulated with sparse posting, high follow/low follower counts, short bios, and numeric-heavy usernames; real profiles get the opposite pattern.

> ⚠️ Because the synthetic classes are generated from clearly separated distributions, the demo dashboard shows near-perfect metrics (AUC ≈ 1.0). This **will not hold on real-world data** — swap in the actual Kaggle CSVs (instructions below) for realistic performance.

---

## 📊 Results (Synthetic Data Demo)

From `analysis_dashboard.png`:

| Metric | Score |
|---|---|
| Accuracy | 100% (synthetic data) |
| ROC-AUC | 1.000 |
| 5-Fold CV ROC-AUC | reported in script output |

**Top features by importance:** `followers` → `posts` → `account_age_days` → `follower_follow_ratio` → `bio_length`

The dashboard includes 11 panels: class distribution, confusion matrix, ROC curve, feature importances, and per-feature distributions split by real/fake (followers, follow ratio, bio length, profile picture presence, post count, account age, predicted probability histogram).

---

## 🚀 Usage

### 1. Run the standalone script
Trains the model, prints evaluation metrics, saves the 11-panel dashboard, and runs 2 demo predictions.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
python fake_profile_detection.py
```

Output:
```
[1/5] Generating dataset...
[2/5] Preprocessing...
[3/5] Training Random Forest...
[4/5] Evaluating...
[5/5] Generating dashboard...
  Dashboard saved → analysis_dashboard.png

  DEMO PREDICTIONS
  Profile 1: FAKE 🚩  (P(fake) = 0.XXX)
  Profile 2: REAL ✅  (P(fake) = 0.XXX)
```

### 2. Explore the Jupyter notebook
Step-by-step version with markdown explanations, EDA plots (feature distributions, correlation heatmap), and an editable profile predictor cell.

```bash
jupyter notebook fake_profile_detection.ipynb
```

### 3. Run the Flask web app
Serves a live prediction API backed by the same model.

```bash
pip install flask numpy pandas scikit-learn
python app.py
# → http://localhost:5000
```

**API endpoint:**

```
POST /predict
Content-Type: application/json

{
  "profile_pic": 0,
  "nums_in_username": 7,
  "fullname_words": 0,
  "fullname_nums": 3,
  "bio_length": 5,
  "external_url": 0,
  "private": 0,
  "posts": 2,
  "followers": 12,
  "follows": 4800,
  "account_age_days": 20
}
```

**Response:**

```json
{
  "label": "FAKE",
  "probability": 91.4,
  "confidence": 91.4,
  "top_features": [
    {"name": "Followers", "score": 25.8},
    {"name": "Posts", "score": 24.1},
    {"name": "Account Age Days", "score": 14.3},
    {"name": "Follower Follow Ratio", "score": 13.0},
    {"name": "Bio Length", "score": 8.4}
  ]
}
```

> ⚠️ **Setup note:** `app.py` calls `render_template("index.html")`, which requires a `templates/index.html` file in the project root. This file isn't included yet — create a simple form/dashboard page that POSTs profile data to `/predict`, or let me know and I can build one for you.

---

## 🛠 Tech Stack

- **Language:** Python 3
- **ML:** scikit-learn (RandomForestClassifier, train_test_split, StratifiedKFold, metrics)
- **Data:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Web:** Flask

---

## 🔄 Using Real Data Instead of Synthetic

Replace the synthetic generator with the actual Kaggle dataset:

```python
# Download from:
# https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-accounts

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
df = pd.concat([train_df, test_df], ignore_index=True)
# Then continue with preprocess() → train_model() → evaluate_model()
```

---

## 📚 Possible Extensions

- Swap synthetic data for the real Kaggle dataset (or Twitter/X, TikTok equivalents)
- Add NLP-based bio/username analysis (e.g., suspicious keyword detection)
- Try gradient boosting (XGBoost/LightGBM) for comparison
- Add SHAP values for richer explainability than built-in feature importances
- Persist the trained model with `joblib` instead of retraining on every Flask startup
- Add authentication + rate limiting to the `/predict` endpoint for production use

---

## ⚠️ Disclaimer

Built for academic/educational purposes. Performance shown is on synthetic data and is not representative of real-world deployment accuracy.

---

*Fake Profile Detection — Random Forest Classifier*
