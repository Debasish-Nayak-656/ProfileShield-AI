"""
Fake Profile Detection - Flask Web App
Run: python app.py
Visit: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ── Train model on startup ──────────────────────────────────────────
def generate_dataset(n_samples=5000, random_state=42):
    np.random.seed(random_state)
    n_fake = n_samples // 2
    n_real = n_samples - n_fake

    def make_profiles(n, fake=False):
        if fake:
            return {
                "profile_pic":      np.random.choice([0, 1], n, p=[0.6, 0.4]),
                "nums_in_username": np.random.randint(2, 10, n),
                "fullname_words":   np.random.randint(0, 2, n),
                "fullname_nums":    np.random.randint(0, 5, n),
                "bio_length":       np.random.randint(0, 30, n),
                "external_url":     np.random.choice([0, 1], n, p=[0.7, 0.3]),
                "private":          np.random.choice([0, 1], n, p=[0.8, 0.2]),
                "posts":            np.random.randint(0, 20, n),
                "followers":        np.random.randint(0, 300, n),
                "follows":          np.random.randint(500, 7500, n),
                "account_age_days": np.random.randint(1, 300, n),
            }
        else:
            return {
                "profile_pic":      np.random.choice([0, 1], n, p=[0.05, 0.95]),
                "nums_in_username": np.random.randint(0, 3, n),
                "fullname_words":   np.random.randint(1, 4, n),
                "fullname_nums":    np.random.randint(0, 2, n),
                "bio_length":       np.random.randint(20, 160, n),
                "external_url":     np.random.choice([0, 1], n, p=[0.4, 0.6]),
                "private":          np.random.choice([0, 1], n, p=[0.5, 0.5]),
                "posts":            np.random.randint(10, 500, n),
                "followers":        np.random.randint(100, 10000, n),
                "follows":          np.random.randint(50, 2000, n),
                "account_age_days": np.random.randint(180, 3000, n),
            }

    fake_df = pd.DataFrame(make_profiles(n_fake, fake=True)); fake_df["label"] = 1
    real_df = pd.DataFrame(make_profiles(n_real, fake=False)); real_df["label"] = 0
    df = pd.concat([fake_df, real_df], ignore_index=True).sample(frac=1, random_state=random_state)
    df["follower_follow_ratio"] = df["followers"] / (df["follows"] + 1)
    df["posts_per_day"]         = df["posts"] / (df["account_age_days"] + 1)
    return df

FEATURES = [
    "profile_pic", "nums_in_username", "fullname_words", "fullname_nums",
    "bio_length", "external_url", "private", "posts", "followers", "follows",
    "account_age_days", "follower_follow_ratio", "posts_per_day"
]

print("Training model...")
df = generate_dataset(5000)
X = df[FEATURES]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_split=5,
    min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)
print("Model ready!")

# ── Routes ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        row = pd.DataFrame([{
            "profile_pic":      int(data["profile_pic"]),
            "nums_in_username": int(data["nums_in_username"]),
            "fullname_words":   int(data["fullname_words"]),
            "fullname_nums":    int(data["fullname_nums"]),
            "bio_length":       int(data["bio_length"]),
            "external_url":     int(data["external_url"]),
            "private":          int(data["private"]),
            "posts":            int(data["posts"]),
            "followers":        int(data["followers"]),
            "follows":          int(data["follows"]),
            "account_age_days": int(data["account_age_days"]),
        }])
        row["follower_follow_ratio"] = row["followers"] / (row["follows"] + 1)
        row["posts_per_day"]         = row["posts"] / (row["account_age_days"] + 1)

        prob  = float(model.predict_proba(row[FEATURES])[0][1])
        label = "FAKE" if prob >= 0.5 else "REAL"

        # Top contributing features
        importances = model.feature_importances_
        top = sorted(zip(FEATURES, importances), key=lambda x: -x[1])[:5]
        top_features = [{"name": f.replace("_", " ").title(), "score": round(s * 100, 1)} for f, s in top]

        return jsonify({
            "label": label,
            "probability": round(prob * 100, 1),
            "confidence": round((prob if prob >= 0.5 else 1 - prob) * 100, 1),
            "top_features": top_features
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
