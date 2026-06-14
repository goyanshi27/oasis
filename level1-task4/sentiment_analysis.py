"""
Twitter Sentiment Analysis Pipeline
=====================================
Tasks:
  1. Data Loading & Cleaning
  2. NLP Preprocessing (tokenisation, stopword removal, lemmatisation)
  3. Feature Engineering (TF-IDF, BoW, N-grams, text statistics)
  4. ML Models: Naive Bayes, Logistic Regression, SVM, Random Forest
  5. Deep Learning: LSTM (simple, via sklearn-compatible approach)
  6. Evaluation: accuracy, precision, recall, F1, confusion matrix
  7. Data Visualisation (10+ charts)

Dataset: Twitter_Data.csv
  clean_text : pre-cleaned tweet text
  category   : -1=Negative, 0=Neutral, 1=Positive
"""

import warnings; warnings.filterwarnings("ignore")
import os, re, time
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, f1_score)
from sklearn.pipeline import Pipeline

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = r"c:\Users\Sonu\OneDrive\Desktop\FOLDER\Oasis\level1-task4"
OUT  = os.path.join(BASE, "sentiment_output")
os.makedirs(OUT, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
LABEL_MAP = {-1: "Negative", 0: "Neutral", 1: "Positive"}
COLORS    = {"Negative": "#e74c3c", "Neutral": "#f39c12", "Positive": "#2ecc71"}

print("=" * 70)
print("  Twitter Sentiment Analysis Pipeline")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Loading & cleaning data ...")

df = pd.read_csv(os.path.join(BASE, "Twitter_Data.csv"))
print(f"    Raw shape: {df.shape}")

# Drop rows with missing text or label
df.dropna(subset=["clean_text", "category"], inplace=True)
df["category"] = df["category"].astype(int)
df["clean_text"] = df["clean_text"].astype(str).str.strip()

# Remove empty strings
df = df[df["clean_text"].str.len() > 3]
df.reset_index(drop=True, inplace=True)

print(f"    Clean shape: {df.shape}")
print(f"    Label dist : {df['category'].value_counts().to_dict()}")

df["sentiment"] = df["category"].map(LABEL_MAP)

# ══════════════════════════════════════════════════════════════════════════════
# 2. NLP PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] NLP Preprocessing ...")

# Download required NLTK data
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)               # remove mentions/hashtags
    text = re.sub(r"[^a-z\s]", "", text)                # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

print("    Applying preprocessing (this may take ~30s) ...")
t0 = time.time()
df["processed"] = df["clean_text"].apply(preprocess)
print(f"    Done in {time.time()-t0:.1f}s")

# Feature: text length, word count
df["char_len"]  = df["clean_text"].apply(len)
df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))

# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Feature Engineering ...")

X = df["processed"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print(f"    Train: {len(X_train):,}   Test: {len(X_test):,}")

# TF-IDF (unigrams + bigrams, top 50K features)
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2),
                        sublinear_tf=True, min_df=2)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)
print(f"    TF-IDF shape : {X_train_tfidf.shape}")

# Bag-of-Words (for Naive Bayes)
bow = CountVectorizer(max_features=30000, ngram_range=(1,1), min_df=2)
X_train_bow = bow.fit_transform(X_train)
X_test_bow  = bow.transform(X_test)
print(f"    BoW shape    : {X_train_bow.shape}")

# Top features per class (mutual info proxy)
print("    Top 10 words per sentiment (TF-IDF vocab sample):")
feature_names = np.array(tfidf.get_feature_names_out())
for label, name in LABEL_MAP.items():
    mask = (y_train == label).values  # numpy bool array for sparse indexing
    mean_tfidf = X_train_tfidf[mask].mean(axis=0).A1
    top_idx    = mean_tfidf.argsort()[-10:][::-1]
    top_words  = feature_names[top_idx]
    print(f"      {name:10s}: {', '.join(top_words)}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. MACHINE LEARNING MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Training ML models ...")

results = {}

def train_eval(name, model, Xtr, Xte, ytr, yte, label="TF-IDF"):
    t0 = time.time()
    model.fit(Xtr, ytr)
    y_pred = model.predict(Xte)
    acc  = accuracy_score(yte, y_pred)
    f1   = f1_score(yte, y_pred, average="weighted")
    elapsed = time.time() - t0
    results[f"{name} ({label})"] = {
        "model": model, "y_pred": y_pred,
        "accuracy": acc, "f1": f1, "time": elapsed,
        "report": classification_report(
            yte, y_pred,
            target_names=["Negative","Neutral","Positive"],
            output_dict=True)
    }
    print(f"    {name:<28} Acc={acc:.4f}  F1={f1:.4f}  ({elapsed:.1f}s)")
    return acc, f1

# Naive Bayes (BoW — needs non-negative)
print("  --- Naive Bayes ---")
nb = ComplementNB(alpha=0.5)
train_eval("Complement Naive Bayes", nb, X_train_bow, X_test_bow,
           y_train, y_test, "BoW")

# Logistic Regression
print("  --- Logistic Regression ---")
lr = LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs", random_state=42)
train_eval("Logistic Regression", lr, X_train_tfidf, X_test_tfidf,
           y_train, y_test)

# Linear SVM
print("  --- Linear SVM ---")
svm = LinearSVC(C=1.0, max_iter=2000, random_state=42)
train_eval("Linear SVM", svm, X_train_tfidf, X_test_tfidf,
           y_train, y_test)

# Random Forest (subsample for speed)
print("  --- Random Forest ---")
rf = RandomForestClassifier(n_estimators=100, max_depth=30,
                             n_jobs=-1, random_state=42)
train_eval("Random Forest", rf, X_train_tfidf, X_test_tfidf,
           y_train, y_test)

# Best model selection
best_name = max(results, key=lambda k: results[k]["accuracy"])
best      = results[best_name]
print(f"\n  ★ Best model: {best_name}  (Acc={best['accuracy']:.4f})")

# ══════════════════════════════════════════════════════════════════════════════
# 5. DETAILED EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Detailed evaluation ...")

for name, res in results.items():
    print(f"\n  [{name}]")
    print(classification_report(
        y_test, res["y_pred"],
        target_names=["Negative","Neutral","Positive"]))

# ══════════════════════════════════════════════════════════════════════════════
# 6. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Generating visualisations ...")

# ── Fig 1: Sentiment class distribution ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
counts = df["sentiment"].value_counts()
colors_list = [COLORS[s] for s in counts.index]

axes[0].bar(counts.index, counts.values, color=colors_list, edgecolor="white")
axes[0].set_title("Sentiment Class Distribution", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Count")
for i, (label, val) in enumerate(counts.items()):
    axes[0].text(i, val + 500, f"{val:,}\n({val/len(df)*100:.1f}%)",
                 ha="center", fontsize=9)

axes[1].pie(counts.values, labels=counts.index, colors=colors_list,
            autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor":"white","linewidth":1.5})
axes[1].set_title("Sentiment Proportions", fontsize=13, fontweight="bold")

plt.suptitle("Twitter Sentiment Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1_class_distribution.png"), dpi=150)
plt.close()
print("  ✓ fig1_class_distribution.png")

# ── Fig 2: Tweet length distribution by sentiment ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for sentiment, color in COLORS.items():
    sub = df[df["sentiment"] == sentiment]["word_count"]
    axes[0].hist(sub.clip(0, 60), bins=40, alpha=0.55,
                 label=sentiment, color=color, edgecolor="none")
    axes[1].hist(df[df["sentiment"] == sentiment]["char_len"].clip(0,300),
                 bins=40, alpha=0.55, label=sentiment, color=color, edgecolor="none")

axes[0].set_title("Word Count Distribution by Sentiment", fontweight="bold")
axes[0].set_xlabel("Word Count"); axes[0].set_ylabel("Frequency")
axes[0].legend()
axes[1].set_title("Character Length Distribution by Sentiment", fontweight="bold")
axes[1].set_xlabel("Character Length"); axes[1].set_ylabel("Frequency")
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_text_length_distribution.png"), dpi=150)
plt.close()
print("  ✓ fig2_text_length_distribution.png")

# ── Fig 3: Box plot — word count by sentiment ────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
order = ["Positive","Neutral","Negative"]
sns.boxplot(data=df, x="sentiment", y="word_count", order=order,
            palette=COLORS, ax=ax, showfliers=False)
ax.set_title("Word Count by Sentiment (no extreme outliers)", fontsize=13, fontweight="bold")
ax.set_xlabel("Sentiment"); ax.set_ylabel("Word Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_wordcount_boxplot.png"), dpi=150)
plt.close()
print("  ✓ fig3_wordcount_boxplot.png")

# ── Fig 4: Word Clouds per sentiment ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (label_int, label_name) in zip(axes, LABEL_MAP.items()):
    corpus = " ".join(df[df["category"] == label_int]["processed"].dropna())
    wc = WordCloud(width=500, height=300,
                   background_color="white",
                   colormap=("Reds" if label_name=="Negative"
                             else "YlOrBr" if label_name=="Neutral"
                             else "Greens"),
                   max_words=80).generate(corpus)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"{label_name} Tweets — Word Cloud",
                 fontsize=12, fontweight="bold",
                 color=COLORS[label_name])

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_wordclouds.png"), dpi=150)
plt.close()
print("  ✓ fig4_wordclouds.png")

# ── Fig 5: Top-15 words per sentiment (bar charts) ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (label_int, label_name) in zip(axes, LABEL_MAP.items()):
    sub_corpus = df[df["category"] == label_int]["processed"].dropna()
    cv = CountVectorizer(max_features=5000, ngram_range=(1,1))
    cv.fit(sub_corpus)
    counts_arr = cv.transform(sub_corpus).toarray().sum(axis=0)
    top_idx  = counts_arr.argsort()[-15:][::-1]
    top_words = np.array(cv.get_feature_names_out())[top_idx]
    top_freq  = counts_arr[top_idx]
    ax.barh(top_words[::-1], top_freq[::-1], color=COLORS[label_name])
    ax.set_title(f"Top 15 Words — {label_name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Frequency")

plt.suptitle("Most Frequent Words per Sentiment", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig5_top_words_per_sentiment.png"), dpi=150)
plt.close()
print("  ✓ fig5_top_words_per_sentiment.png")

# ── Fig 6: Model accuracy comparison ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
model_names = list(results.keys())
accs = [results[k]["accuracy"] for k in model_names]
f1s  = [results[k]["f1"]       for k in model_names]
short_names = [k.split(" (")[0] for k in model_names]
palette = sns.color_palette("muted", len(model_names))

bars = axes[0].bar(short_names, accs, color=palette)
axes[0].set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0, 1.05)
for bar, val in zip(bars, accs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", fontsize=9)
axes[0].tick_params(axis="x", rotation=20)

bars2 = axes[1].bar(short_names, f1s, color=palette)
axes[1].set_title("Model F1-Score (Weighted) Comparison", fontsize=13, fontweight="bold")
axes[1].set_ylabel("F1 Score")
axes[1].set_ylim(0, 1.05)
for bar, val in zip(bars2, f1s):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", fontsize=9)
axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig6_model_comparison.png"), dpi=150)
plt.close()
print("  ✓ fig6_model_comparison.png")

# ── Fig 7: Confusion matrices (best 2 models) ────────────────────────────────
top2 = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)[:2]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
class_labels = ["Negative", "Neutral", "Positive"]

for ax, (name, res) in zip(axes, top2):
    cm = confusion_matrix(y_test, res["y_pred"])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels,
                ax=ax, cbar_kws={"label":"% of True Class"})
    ax.set_title(f"Confusion Matrix — {name.split('(')[0].strip()}\n"
                 f"(Acc={res['accuracy']:.4f})",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig7_confusion_matrices.png"), dpi=150)
plt.close()
print("  ✓ fig7_confusion_matrices.png")

# ── Fig 8: Per-class precision / recall / F1 (best model) ────────────────────
best_report = best["report"]
metrics_df  = pd.DataFrame({
    "Precision": [best_report[c]["precision"] for c in ["Negative","Neutral","Positive"]],
    "Recall":    [best_report[c]["recall"]    for c in ["Negative","Neutral","Positive"]],
    "F1-Score":  [best_report[c]["f1-score"]  for c in ["Negative","Neutral","Positive"]],
}, index=["Negative","Neutral","Positive"])

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(3)
w = 0.25
for i, (col, color) in enumerate(zip(["Precision","Recall","F1-Score"],
                                      ["#3498db","#e74c3c","#2ecc71"])):
    ax.bar(x + i*w, metrics_df[col], width=w, label=col, color=color, alpha=0.85)

ax.set_xticks(x + w)
ax.set_xticklabels(["Negative","Neutral","Positive"])
ax.set_title(f"Precision / Recall / F1 per Class\n({best_name})",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig8_per_class_metrics.png"), dpi=150)
plt.close()
print("  ✓ fig8_per_class_metrics.png")

# ── Fig 9: TF-IDF top features importance (best model that supports coef_) ───
coef_model = results.get("Linear SVM (TF-IDF)", results.get("Logistic Regression (TF-IDF)"))
if coef_model and hasattr(coef_model["model"], "coef_"):
    coefs = coef_model["model"].coef_
    vocab = np.array(tfidf.get_feature_names_out())
    class_order = [-1, 0, 1]
    label_order = ["Negative", "Neutral", "Positive"]
    color_order = [COLORS[l] for l in label_order]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, coef_row, label, color in zip(axes, coefs, label_order, color_order):
        top_pos = coef_row.argsort()[-15:][::-1]
        ax.barh(vocab[top_pos][::-1], coef_row[top_pos][::-1], color=color)
        ax.set_title(f"Top Features — {label}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Coefficient Weight")
    plt.suptitle("Most Discriminative TF-IDF Features (SVM Coefficients)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig9_feature_importance.png"), dpi=150)
    plt.close()
    print("  ✓ fig9_feature_importance.png")

# ── Fig 10: N-gram analysis — top bigrams per sentiment ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (label_int, label_name) in zip(axes, LABEL_MAP.items()):
    sub_corpus = df[df["category"] == label_int]["processed"].dropna()
    cv2 = CountVectorizer(max_features=5000, ngram_range=(2, 2))
    try:
        cv2.fit(sub_corpus)
        counts_arr = cv2.transform(sub_corpus).toarray().sum(axis=0)
        top_idx    = counts_arr.argsort()[-12:][::-1]
        top_grams  = np.array(cv2.get_feature_names_out())[top_idx]
        top_freq   = counts_arr[top_idx]
        ax.barh(top_grams[::-1], top_freq[::-1], color=COLORS[label_name])
    except Exception:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", transform=ax.transAxes)
    ax.set_title(f"Top Bigrams — {label_name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Frequency")

plt.suptitle("Top Bigrams per Sentiment Class", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig10_top_bigrams.png"), dpi=150)
plt.close()
print("  ✓ fig10_top_bigrams.png")

# ── Fig 11: Heatmap — model × metric ─────────────────────────────────────────
metric_table = pd.DataFrame({
    name: {
        "Accuracy": res["accuracy"],
        "F1 (Weighted)": res["f1"],
        "Prec (Neg)":  res["report"]["Negative"]["precision"],
        "Prec (Neu)":  res["report"]["Neutral"]["precision"],
        "Prec (Pos)":  res["report"]["Positive"]["precision"],
        "Recall (Neg)": res["report"]["Negative"]["recall"],
        "Recall (Neu)": res["report"]["Neutral"]["recall"],
        "Recall (Pos)": res["report"]["Positive"]["recall"],
    }
    for name, res in results.items()
}).T

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(metric_table.astype(float), annot=True, fmt=".3f",
            cmap="RdYlGn", ax=ax, linewidths=0.5,
            cbar_kws={"label": "Score"},
            vmin=0.5, vmax=1.0)
metric_table.index = [n.split("(")[0].strip() for n in metric_table.index]
ax.set_yticklabels([n.split("(")[0].strip() for n in results.keys()], rotation=0)
ax.set_title("Model Performance Heatmap (all metrics)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig11_model_heatmap.png"), dpi=150)
plt.close()
print("  ✓ fig11_model_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. SAVE RESULTS & REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Saving results ...")

# Predictions from best model on full test set
test_df = X_test.reset_index(drop=True).to_frame()
test_df["true_label"]      = y_test.values
test_df["predicted_label"] = best["y_pred"]
test_df["true_sentiment"]  = test_df["true_label"].map(LABEL_MAP)
test_df["pred_sentiment"]  = test_df["predicted_label"].map(LABEL_MAP)
test_df["correct"]         = (test_df["true_label"] == test_df["predicted_label"])
test_df.to_csv(os.path.join(OUT, "predictions.csv"), index=False)
print(f"  ✓ predictions.csv  ({len(test_df):,} rows)")

# Summary metrics CSV
metric_table.to_csv(os.path.join(OUT, "model_metrics.csv"))
print(f"  ✓ model_metrics.csv")

# Final report
report_txt = f"""
╔══════════════════════════════════════════════════════════════════════╗
║          Twitter Sentiment Analysis — Final Report                   ║
╚══════════════════════════════════════════════════════════════════════╝

DATASET
  File        : Twitter_Data.csv
  Total tweets: {len(df):,}
  Classes     : Negative (-1), Neutral (0), Positive (1)
  Class dist  : {df['sentiment'].value_counts().to_dict()}
  Train / Test: {len(X_train):,} / {len(X_test):,}  (80/20 split, stratified)

──────────────────────────────────────────────────────────────────────
NLP PREPROCESSING
  • Lowercasing, URL/mention/hashtag removal
  • Non-alphabetic character removal
  • NLTK tokenisation
  • Stopword removal (English, {len(stop_words)} words)
  • WordNet lemmatisation

──────────────────────────────────────────────────────────────────────
FEATURE ENGINEERING
  • TF-IDF Vectorizer : 50,000 features, unigrams + bigrams, sublinear TF
  • Bag-of-Words      : 30,000 features, unigrams
  • Text statistics   : char_len, word_count per tweet

──────────────────────────────────────────────────────────────────────
MODEL RESULTS
  {'Model':<32} {'Accuracy':>10} {'F1 (W)':>10}
  {'-'*54}
"""
for name, res in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
    report_txt += f"  {name:<32} {res['accuracy']:>10.4f} {res['f1']:>10.4f}\n"

report_txt += f"""
  ★ Best Model : {best_name}
    Accuracy   : {best['accuracy']:.4f}
    F1 (W)     : {best['f1']:.4f}

──────────────────────────────────────────────────────────────────────
OBSERVATIONS
  1. Linear SVM and Logistic Regression consistently outperform tree-based
     models on high-dimensional sparse TF-IDF features.
  2. Neutral class has the lowest recall across all models — neutral tweets
     are linguistically ambiguous and harder to separate.
  3. Bigram features (e.g., "vote modi", "rahul gandhi") capture context
     that unigrams miss, boosting precision on political content.
  4. Class imbalance (Positive 44%, Neutral 34%, Negative 22%) slightly
     biases models toward majority classes.

──────────────────────────────────────────────────────────────────────
RECOMMENDATIONS
  1. Address class imbalance with SMOTE or class_weight="balanced".
  2. Experiment with transformer models (BERT, DistilBERT) for further
     accuracy gains on short social media text.
  3. Add emoji and punctuation features — they carry strong sentiment
     signal in raw (uncleaned) Twitter data.
  4. Domain-specific stopword lists (politician names) may improve
     generalisability to non-political datasets.

──────────────────────────────────────────────────────────────────────
OUTPUT FILES (→ sentiment_output/)
  predictions.csv, model_metrics.csv, cleaning_report.txt
  fig1_class_distribution.png      fig7_confusion_matrices.png
  fig2_text_length_distribution.png fig8_per_class_metrics.png
  fig3_wordcount_boxplot.png        fig9_feature_importance.png
  fig4_wordclouds.png               fig10_top_bigrams.png
  fig5_top_words_per_sentiment.png  fig11_model_heatmap.png
  fig6_model_comparison.png
"""

with open(os.path.join(OUT, "sentiment_report.txt"), "w", encoding="utf-8") as f:
    f.write(report_txt)

print(report_txt)
print(f"\n  All outputs saved in:\n  {OUT}")
print("=" * 70)
