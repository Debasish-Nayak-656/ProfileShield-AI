"""
Fake Profile Detection in Social Media
Using Random Forest Classifier
Dataset: Synthetic (mimics Kaggle Social Media Dataset structure)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC DATASET
# ─────────────────────────────────────────────

def generate_dataset(n_samples=5000, random_state=42):
    """
    Generate a synthetic social media profile dataset.
    Features are designed to mimic real Kaggle fake-profile datasets.
    """
    np.random.seed(random_state)
    n_fake = n_samples // 2
    n_real = n_samples - n_fake

    def make_profiles(n, fake=False):
        if fake:
            return {
                "profile_pic":        np.random.choice([0, 1], n, p=[0.6, 0.4]),
                "nums_in_username":   np.random.randint(2, 10, n),
                "fullname_words":     np.random.randint(0, 2, n),
                "fullname_nums":      np.random.randint(0, 5, n),
                "bio_length":         np.random.randint(0, 30, n),
                "external_url":       np.random.choice([0, 1], n, p=[0.7, 0.3]),
                "private":            np.random.choice([0, 1], n, p=[0.8, 0.2]),
                "posts":              np.random.randint(0, 20, n),
                "followers":          np.random.randint(0, 300, n),
                "follows":            np.random.randint(500, 7500, n),
                "account_age_days":   np.random.randint(1, 300, n),
            }
        else:
            return {
                "profile_pic":        np.random.choice([0, 1], n, p=[0.05, 0.95]),
                "nums_in_username":   np.random.randint(0, 3, n),
                "fullname_words":     np.random.randint(1, 4, n),
                "fullname_nums":      np.random.randint(0, 2, n),
                "bio_length":         np.random.randint(20, 160, n),
                "external_url":       np.random.choice([0, 1], n, p=[0.4, 0.6]),
                "private":            np.random.choice([0, 1], n, p=[0.5, 0.5]),
                "posts":              np.random.randint(10, 500, n),
                "followers":          np.random.randint(100, 10000, n),
                "follows":            np.random.randint(50, 2000, n),
                "account_age_days":   np.random.randint(180, 3000, n),
            }

    fake_df = pd.DataFrame(make_profiles(n_fake, fake=True))
    fake_df["label"] = 1

    real_df = pd.DataFrame(make_profiles(n_real, fake=False))
    real_df["label"] = 0

    df = pd.concat([fake_df, real_df], ignore_index=True).sample(frac=1, random_state=random_state)

    # Derived features
    df["follower_follow_ratio"] = df["followers"] / (df["follows"] + 1)
    df["posts_per_day"] = df["posts"] / (df["account_age_days"] + 1)

    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df):
    FEATURES = [
        "profile_pic", "nums_in_username", "fullname_words", "fullname_nums",
        "bio_length", "external_url", "private", "posts", "followers", "follows",
        "account_age_days", "follower_follow_ratio", "posts_per_day"
    ]
    X = df[FEATURES]
    y = df["label"]
    return X, y, FEATURES


# ─────────────────────────────────────────────
# 3. TRAIN MODEL
# ─────────────────────────────────────────────

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, feature_names):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("=" * 55)
    print("  FAKE PROFILE DETECTION — MODEL EVALUATION")
    print("=" * 55)
    print(f"\n  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    return y_pred, y_proba


# ─────────────────────────────────────────────
# 5. PLOTS
# ─────────────────────────────────────────────

COLORS = {"fake": "#E24B4A", "real": "#1D9E75", "neutral": "#534AB7",
          "bar": "#3266ad", "bg": "#F8F8F7"}

def plot_all(model, X_test, y_test, y_pred, y_proba, feature_names, df):
    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor(COLORS["bg"])
    plt.suptitle("Fake Profile Detection — Full Analysis Dashboard",
                 fontsize=18, fontweight="bold", y=0.98, color="#2C2C2A")

    axes = [
        fig.add_subplot(4, 3, i+1) for i in range(11)
    ]
    for ax in axes:
        ax.set_facecolor(COLORS["bg"])
        for spine in ax.spines.values():
            spine.set_edgecolor("#D3D1C7")
            spine.set_linewidth(0.7)

    # ── 1. Class distribution ──
    ax = axes[0]
    counts = df["label"].value_counts()
    bars = ax.bar(["Real (0)", "Fake (1)"], [counts[0], counts[1]],
                  color=[COLORS["real"], COLORS["fake"]], width=0.5, edgecolor="white")
    for bar, val in zip(bars, [counts[0], counts[1]]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"{val:,}", ha="center", fontsize=10, color="#444441")
    ax.set_title("Class Distribution", fontweight="bold", color="#2C2C2A")
    ax.set_ylabel("Count")

    # ── 2. Confusion Matrix ──
    ax = axes[1]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn", ax=ax,
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
                linewidths=0.5, cbar=False, annot_kws={"size": 13})
    ax.set_title("Confusion Matrix", fontweight="bold", color="#2C2C2A")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    # ── 3. ROC Curve ──
    ax = axes[2]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, color=COLORS["neutral"], lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.4)
    ax.fill_between(fpr, tpr, alpha=0.08, color=COLORS["neutral"])
    ax.set_title("ROC Curve", fontweight="bold", color="#2C2C2A")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(framealpha=0.4)

    # ── 4. Feature Importances ──
    ax = axes[3]
    importances = model.feature_importances_
    idx = np.argsort(importances)
    colors_bar = [COLORS["fake"] if importances[i] > np.median(importances) else COLORS["bar"]
                  for i in idx]
    ax.barh([feature_names[i] for i in idx], importances[idx],
            color=colors_bar, edgecolor="white")
    ax.set_title("Feature Importances", fontweight="bold", color="#2C2C2A")
    ax.set_xlabel("Importance")

    # ── 5. Followers distribution ──
    ax = axes[4]
    for label, color, name in [(1, COLORS["fake"], "Fake"), (0, COLORS["real"], "Real")]:
        data = df[df["label"] == label]["followers"].clip(upper=3000)
        ax.hist(data, bins=40, alpha=0.6, color=color, label=name, edgecolor="white")
    ax.set_title("Followers Distribution", fontweight="bold", color="#2C2C2A")
    ax.set_xlabel("Followers"); ax.set_ylabel("Count"); ax.legend()

    # ── 6. Follower/Following ratio ──
    ax = axes[5]
    for label, color, name in [(1, COLORS["fake"], "Fake"), (0, COLORS["real"], "Real")]:
        data = df[df["label"] == label]["follower_follow_ratio"].clip(upper=20)
        ax.hist(data, bins=40, alpha=0.6, color=color, label=name, edgecolor="white")
    ax.set_title("Follower/Follow Ratio", fontweight="bold", color="#2C2C2A")
    ax.set_xlabel("Ratio"); ax.set_ylabel("Count"); ax.legend()

    # ── 7. Bio length ──
    ax = axes[6]
    for label, color, name in [(1, COLORS["fake"], "Fake"), (0, COLORS["real"], "Real")]:
        data = df[df["label"] == label]["bio_length"]
        ax.hist(data, bins=30, alpha=0.6, color=color, label=name, edgecolor="white")
    ax.set_title("Bio Length Distribution", fontweight="bold", color="#2C2C2A")
    ax.set_xlabel("Bio Length (chars)"); ax.set_ylabel("Count"); ax.legend()

    # ── 8. Profile pic vs label ──
    ax = axes[7]
    cross = df.groupby(["profile_pic", "label"]).size().unstack(fill_value=0)
    cross.plot(kind="bar", ax=ax, color=[COLORS["real"], COLORS["fake"]],
               edgecolor="white", width=0.5)
    ax.set_title("Profile Pic vs Label", fontweight="bold", color="#2C2C2A")
    ax.set_xlabel("Has Profile Pic (0=No, 1=Yes)")
    ax.set_xticklabels(["No Photo", "Has Photo"], rotation=0)
    ax.legend(["Real", "Fake"])

    # ── 9. Posts distribution ──
    ax = axes[8]
    for label, color, name in [(1, COLORS["fake"], "Fake"), (0, COLORS["real"], "Real")]:
        data = df[df["label"] == label]["posts"].clip(upper=300)
        ax.hist(data, bins=30, alpha=0.6, color=color, label=name, edgecolor="white")
    ax.set_title("Post Count Distribution", fontweight="bold", color="#2C2C2A")
    ax.set_xlabel("Posts"); ax.set_ylabel("Count"); ax.legend()

    # ── 10. Account age ──
    ax = axes[9]
    for label, color, name in [(1, COLORS["fake"], "Fake"), (0, COLORS["real"], "Real")]:
        data = df[df["label"] == label]["account_age_days"]
        ax.hist(data, bins=30, alpha=0.6, color=color, label=name, edgecolor="white")
    ax.set_title("Account Age Distribution", fontweight="bold", color="#2C2C2A")
    ax.set_xlabel("Days"); ax.set_ylabel("Count"); ax.legend()

    # ── 11. Prediction probability histogram ──
    ax = axes[10]
    ax.hist(y_proba[y_test == 0], bins=30, alpha=0.6, color=COLORS["real"],
            label="Real", edgecolor="white")
    ax.hist(y_proba[y_test == 1], bins=30, alpha=0.6, color=COLORS["fake"],
            label="Fake", edgecolor="white")
    ax.axvline(0.5, color="#2C2C2A", linestyle="--", lw=1.2, label="Threshold=0.5")
    ax.set_title("Predicted Probability Distribution", fontweight="bold", color="#2C2C2A")
    ax.set_xlabel("P(Fake)"); ax.set_ylabel("Count"); ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("analysis_dashboard.png", dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close()
    print("\n  Dashboard saved → analysis_dashboard.png")


# ─────────────────────────────────────────────
# 6. PREDICT NEW PROFILE (Demo)
# ─────────────────────────────────────────────

def predict_profile(model, feature_names):
    """Demo: predict two sample profiles."""
    samples = pd.DataFrame([
        # Likely fake: no pic, lots of numbers in username, few posts, many follows
        {"profile_pic": 0, "nums_in_username": 7, "fullname_words": 0, "fullname_nums": 3,
         "bio_length": 5, "external_url": 0, "private": 0, "posts": 2,
         "followers": 12, "follows": 4800, "account_age_days": 20},
        # Likely real: has pic, normal username, good history
        {"profile_pic": 1, "nums_in_username": 1, "fullname_words": 2, "fullname_nums": 0,
         "bio_length": 95, "external_url": 1, "private": 0, "posts": 187,
         "followers": 2340, "follows": 410, "account_age_days": 1200},
    ])
    samples["follower_follow_ratio"] = samples["followers"] / (samples["follows"] + 1)
    samples["posts_per_day"]         = samples["posts"] / (samples["account_age_days"] + 1)

    probs = model.predict_proba(samples[feature_names])[:, 1]
    labels = ["FAKE 🚩" if p >= 0.5 else "REAL ✅" for p in probs]

    print("\n  DEMO PREDICTIONS")
    print("  " + "-"*45)
    for i, (label, prob) in enumerate(zip(labels, probs)):
        print(f"  Profile {i+1}: {label}  (P(fake) = {prob:.3f})")
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[1/5] Generating dataset...")
    df = generate_dataset(n_samples=5000)
    print(f"      {len(df):,} profiles | {df['label'].sum()} fake, {(df['label']==0).sum()} real")

    print("[2/5] Preprocessing...")
    X, y, feature_names = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("[3/5] Training Random Forest...")
    model = train_model(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"      5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print("[4/5] Evaluating...\n")
    y_pred, y_proba = evaluate_model(model, X_test, y_test, feature_names)

    print("[5/5] Generating dashboard...")
    plot_all(model, X_test, y_test, y_pred, y_proba, feature_names, df)

    predict_profile(model, feature_names)
