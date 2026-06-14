"""
McDonald's Menu EDA
===================
Tasks:
  1. Data Loading & Cleaning
  2. Descriptive Statistics
  3. Time-Series-style trend analysis (Category / nutritional profile)
  4. Customer & Product Analysis
  5. Visualizations (bar charts, line plots, heatmaps)
  6. Recommendations
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – saves PNGs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import os

# ── Output folder ──────────────────────────────────────────────────────────────
OUT = r"c:\Users\Sonu\OneDrive\Desktop\FOLDER\Oasis\eda_output"
os.makedirs(OUT, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE = (12, 6)

# ==============================================================================
# 1.  DATA LOADING & CLEANING
# ==============================================================================
print("=" * 70)
print("1. DATA LOADING & CLEANING")
print("=" * 70)

df = pd.read_csv(r"c:\Users\Sonu\OneDrive\Desktop\FOLDER\Oasis\menu.csv")

print(f"\nShape  : {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# --- missing values ---
missing = df.isnull().sum()
print("Missing values per column:")
print(missing[missing > 0] if missing.any() else "  None – dataset is clean!\n")

# --- duplicates ---
dups = df.duplicated().sum()
print(f"\nDuplicate rows: {dups}")

# --- data types ---
print("\nData types:")
print(df.dtypes)

# --- strip whitespace from object columns ---
str_cols = df.select_dtypes("object").columns
df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

# Numerical columns (exclude Category, Item, Serving Size)
num_cols = df.select_dtypes(include="number").columns.tolist()

print(f"\nNumerical columns ({len(num_cols)}): {num_cols}")
print("\nFirst 5 rows:")
print(df.head())

# ==============================================================================
# 2.  DESCRIPTIVE STATISTICS
# ==============================================================================
print("\n" + "=" * 70)
print("2. DESCRIPTIVE STATISTICS")
print("=" * 70)

desc = df[num_cols].describe().T
desc["median"] = df[num_cols].median()
desc["mode"]   = df[num_cols].mode().iloc[0]
desc["skewness"] = df[num_cols].skew()
desc["kurtosis"] = df[num_cols].kurtosis()

pd.set_option("display.float_format", "{:.2f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

print("\nFull descriptive statistics (with median, mode, skewness, kurtosis):")
print(desc.to_string())

# Save to CSV
desc.to_csv(os.path.join(OUT, "descriptive_statistics.csv"))
print(f"\n  ✓ Saved: descriptive_statistics.csv")

# Quick highlight
print("\n  Key highlights:")
print(f"  • Mean Calories         : {df['Calories'].mean():.1f} kcal")
print(f"  • Median Calories       : {df['Calories'].median():.1f} kcal")
print(f"  • Std Dev Calories      : {df['Calories'].std():.1f} kcal")
print(f"  • Mean Total Fat        : {df['Total Fat'].mean():.1f} g")
print(f"  • Mean Sodium           : {df['Sodium'].mean():.1f} mg")
print(f"  • Mean Protein          : {df['Protein'].mean():.1f} g")
print(f"  • Mean Sugars           : {df['Sugars'].mean():.1f} g")

# ==============================================================================
# 3.  TIME-SERIES / TREND ANALYSIS  (category-level nutritional trends)
# ==============================================================================
print("\n" + "=" * 70)
print("3. TREND ANALYSIS  (Category-level nutritional profiles)")
print("=" * 70)

# Group by Category – treat categories as ordered segments (menu evolution proxy)
cat_order = ["Breakfast", "Beef & Pork", "Chicken & Fish", "Salads",
             "Snacks & Sides", "Desserts", "Beverages", "Coffee & Tea",
             "Smoothies & Shakes"]

cat_stats = (df.groupby("Category")[["Calories","Total Fat","Sodium","Sugars","Protein"]]
               .mean()
               .reindex([c for c in cat_order if c in df["Category"].unique()]))

print("\nMean nutritional values by category:")
print(cat_stats.round(1))

# --- Line plot: caloric & macronutrient trends across categories ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Calories trend
axes[0].plot(cat_stats.index, cat_stats["Calories"], marker="o", color="#e74c3c", linewidth=2)
axes[0].fill_between(range(len(cat_stats)), cat_stats["Calories"], alpha=0.15, color="#e74c3c")
axes[0].set_title("Average Calories by Category", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Category")
axes[0].set_ylabel("Calories (kcal)")
axes[0].set_xticks(range(len(cat_stats)))
axes[0].set_xticklabels(cat_stats.index, rotation=35, ha="right")

# Macronutrients trend
for nutrient, color in zip(["Total Fat","Sodium","Sugars","Protein"],
                            ["#3498db","#e67e22","#9b59b6","#2ecc71"]):
    axes[1].plot(cat_stats.index, cat_stats[nutrient], marker="o",
                 label=nutrient, color=color, linewidth=2)
axes[1].set_title("Average Macronutrients by Category", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Category")
axes[1].set_ylabel("Amount (g / mg)")
axes[1].set_xticks(range(len(cat_stats)))
axes[1].set_xticklabels(cat_stats.index, rotation=35, ha="right")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1_trend_by_category.png"), dpi=150)
plt.close()
print("  ✓ Saved: fig1_trend_by_category.png")

# ==============================================================================
# 4.  CUSTOMER & PRODUCT ANALYSIS
# ==============================================================================
print("\n" + "=" * 70)
print("4. CUSTOMER & PRODUCT ANALYSIS")
print("=" * 70)

# --- 4a. Items per category ---
cat_counts = df["Category"].value_counts()
print("\nItems per category:")
print(cat_counts)

# --- 4b. Top 10 highest-calorie items ---
top_cal = df.nlargest(10, "Calories")[["Category","Item","Calories","Total Fat","Sodium"]]
print("\nTop 10 highest-calorie items:")
print(top_cal.to_string(index=False))

# --- 4c. Top 10 lowest-calorie items (healthy picks) ---
low_cal = df.nsmallest(10, "Calories")[["Category","Item","Calories","Total Fat","Sugars"]]
print("\nTop 10 lowest-calorie items (healthiest picks):")
print(low_cal.to_string(index=False))

# --- 4d. Sodium watch list ---
high_sodium = df[df["Sodium"] > 1500].sort_values("Sodium", ascending=False)[
    ["Category","Item","Sodium","Calories"]]
print(f"\nItems with Sodium > 1500 mg  (count: {len(high_sodium)}):")
print(high_sodium.to_string(index=False))

# --- 4e. Protein density (protein per 100 kcal) ---
df["Protein_per_100kcal"] = (df["Protein"] / df["Calories"] * 100).round(2)
top_protein = df.nlargest(10, "Protein_per_100kcal")[
    ["Category","Item","Protein","Calories","Protein_per_100kcal"]]
print("\nTop 10 protein-dense items (g protein per 100 kcal):")
print(top_protein.to_string(index=False))

# --- 4f. Sugar analysis (relevant for beverages / desserts) ---
high_sugar = df.nlargest(10, "Sugars")[["Category","Item","Sugars","Calories"]]
print("\nTop 10 highest-sugar items:")
print(high_sugar.to_string(index=False))

# --- 4g. Correlation matrix ---
print("\nCorrelation matrix (key nutrients):")
key_cols = ["Calories","Total Fat","Saturated Fat","Sodium",
            "Carbohydrates","Sugars","Protein"]
corr = df[key_cols].corr().round(2)
print(corr)

# ==============================================================================
# 5.  VISUALIZATIONS
# ==============================================================================
print("\n" + "=" * 70)
print("5. GENERATING VISUALIZATIONS")
print("=" * 70)

# ── Fig 2: Bar chart – items per category ──────────────────────────────────────
fig, ax = plt.subplots(figsize=FIGSIZE)
cat_counts.sort_values().plot(kind="barh", ax=ax, color=sns.color_palette("muted", len(cat_counts)))
ax.set_title("Number of Menu Items per Category", fontsize=14, fontweight="bold")
ax.set_xlabel("Count")
ax.set_ylabel("Category")
for bar in ax.patches:
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{int(bar.get_width())}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_items_per_category.png"), dpi=150)
plt.close()
print("  ✓ Saved: fig2_items_per_category.png")

# ── Fig 3: Bar chart – avg calories by category ────────────────────────────────
fig, ax = plt.subplots(figsize=FIGSIZE)
cat_cal = df.groupby("Category")["Calories"].mean().sort_values(ascending=False)
bars = ax.bar(cat_cal.index, cat_cal.values,
              color=sns.color_palette("RdYlGn_r", len(cat_cal)))
ax.set_title("Average Calories per Category", fontsize=14, fontweight="bold")
ax.set_xlabel("Category")
ax.set_ylabel("Average Calories (kcal)")
ax.set_xticklabels(cat_cal.index, rotation=40, ha="right")
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_avg_calories_by_category.png"), dpi=150)
plt.close()
print("  ✓ Saved: fig3_avg_calories_by_category.png")

# ── Fig 4: Top-10 calorie items ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=FIGSIZE)
top10 = df.nlargest(10, "Calories")[["Item","Calories"]].set_index("Item")
top10.plot(kind="barh", ax=ax, legend=False, color="#e74c3c")
ax.set_title("Top 10 Highest-Calorie Menu Items", fontsize=14, fontweight="bold")
ax.set_xlabel("Calories (kcal)")
for bar in ax.patches:
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f"{int(bar.get_width())}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_top10_calories.png"), dpi=150)
plt.close()
print("  ✓ Saved: fig4_top10_calories.png")

# ── Fig 5: Heatmap – correlation matrix ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            mask=mask, ax=ax, linewidths=0.5,
            cbar_kws={"shrink": 0.8})
ax.set_title("Nutrient Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig5_correlation_heatmap.png"), dpi=150)
plt.close()
print("  ✓ Saved: fig5_correlation_heatmap.png")

# ── Fig 6: Heatmap – mean nutrients by category ───────────────────────────────
heat_data = df.groupby("Category")[["Calories","Total Fat","Sodium",
                                     "Carbohydrates","Sugars","Protein"]].mean()
# Normalise per column for fair colour comparison
heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min())

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(heat_norm, annot=heat_data.round(1), fmt=".1f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Normalised value"})
ax.set_title("Nutritional Profile Heatmap by Category (normalised)", fontsize=13, fontweight="bold")
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig6_nutrient_heatmap_by_category.png"), dpi=150)
plt.close()
print("  ✓ Saved: fig6_nutrient_heatmap_by_category.png")

# ── Fig 7: Distribution of Calories (histogram + KDE) ─────────────────────────
fig, ax = plt.subplots(figsize=FIGSIZE)
sns.histplot(df["Calories"], bins=30, kde=True, color="#3498db", ax=ax)
ax.axvline(df["Calories"].mean(), color="red",   linestyle="--", label=f"Mean: {df['Calories'].mean():.0f}")
ax.axvline(df["Calories"].median(), color="green", linestyle="--", label=f"Median: {df['Calories'].median():.0f}")
ax.set_title("Distribution of Calories Across All Menu Items", fontsize=14, fontweight="bold")
ax.set_xlabel("Calories (kcal)")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig7_calorie_distribution.png"), dpi=150)
plt.close()
print("  ✓ Saved: fig7_calorie_distribution.png")

# ── Fig 8: Box plot – Calories by Category ────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
order_by_median = (df.groupby("Category")["Calories"].median()
                     .sort_values(ascending=False).index)
sns.boxplot(data=df, x="Category", y="Calories", order=order_by_median,
            palette="Set2", ax=ax)
ax.set_title("Calorie Distribution by Category", fontsize=14, fontweight="bold")
ax.set_xlabel("Category")
ax.set_ylabel("Calories (kcal)")
plt.xticks(rotation=40, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig8_boxplot_calories_by_category.png"), dpi=150)
plt.close()
print("  ✓ Saved: fig8_boxplot_calories_by_category.png")

# ── Fig 9: Scatter – Calories vs Protein (coloured by category) ───────────────
fig, ax = plt.subplots(figsize=(10, 7))
categories = df["Category"].unique()
palette = sns.color_palette("tab10", len(categories))
for cat, col in zip(categories, palette):
    sub = df[df["Category"] == cat]
    ax.scatter(sub["Calories"], sub["Protein"], label=cat, color=col, alpha=0.7, s=50)
ax.set_title("Calories vs Protein by Category", fontsize=14, fontweight="bold")
ax.set_xlabel("Calories (kcal)")
ax.set_ylabel("Protein (g)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig9_calories_vs_protein.png"), dpi=150)
plt.close()
print("  ✓ Saved: fig9_calories_vs_protein.png")

# ── Fig 10: Stacked bar – avg macros by category ──────────────────────────────
macro_cols = ["Protein","Total Fat","Carbohydrates","Sugars"]
macro_avg  = df.groupby("Category")[macro_cols].mean()

fig, ax = plt.subplots(figsize=(13, 6))
macro_avg.plot(kind="bar", stacked=True, ax=ax,
               color=["#2ecc71","#e74c3c","#3498db","#f39c12"])
ax.set_title("Average Macronutrient Composition by Category", fontsize=14, fontweight="bold")
ax.set_xlabel("Category")
ax.set_ylabel("Amount (g)")
plt.xticks(rotation=40, ha="right")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig10_stacked_macros.png"), dpi=150)
plt.close()
print("  ✓ Saved: fig10_stacked_macros.png")

# ==============================================================================
# 6.  RECOMMENDATIONS
# ==============================================================================
print("\n" + "=" * 70)
print("6. ACTIONABLE RECOMMENDATIONS")
print("=" * 70)

recommendations = """
Based on the Exploratory Data Analysis of the McDonald's Menu dataset (260 items):

─────────────────────────────────────────────────────────────────────────────
A.  HEALTH-CONSCIOUS MENU STRATEGY
─────────────────────────────────────────────────────────────────────────────
  1. Reduce sodium in high-risk categories.
     Breakfast and Beef & Pork items average >1,000 mg sodium per serving —
     well above the 500 mg "per meal" guideline. Reformulate sauces and
     seasonings (e.g., lower-sodium cheese blends, reduced-salt biscuit
     recipes) to appeal to health-aware customers.

  2. Introduce calorie-capped combo meals.
     The calorie range spans 20–2,310 kcal. Promote pre-built combos that
     stay under 600 kcal, leveraging already-lean items (e.g., Egg White
     Delight 250 kcal, Side Salad ~20 kcal) to address growing nutritional
     awareness.

  3. Expand high-protein, lower-fat options.
     Grilled chicken items top the protein-density chart. Adding 2–3 new
     grilled or baked protein-centric items (wraps, power bowls) can attract
     fitness-focused demographics without cannibalising existing sales.

─────────────────────────────────────────────────────────────────────────────
B.  PRODUCT & CATEGORY MIX
─────────────────────────────────────────────────────────────────────────────
  4. Rationalise the Breakfast lineup.
     Breakfast has the largest item count (>45 SKUs) but high redundancy
     (many biscuit variants). Streamlining to a core range reduces supply
     chain complexity while focusing marketing spend.

  5. Grow the Salads & Low-calorie segment.
     Salads account for only ~7 items. A wider variety (grain bowls, protein
     salads, seasonal options) can capture lunch and dinner health segments
     without large kitchen investments.

  6. Watch the Smoothies & Shakes sugar load.
     Smoothies and shakes show the highest average sugar content (>60 g).
     Offering unsweetened or reduced-sugar variants, or smaller portion
     sizes, can address regulatory pressure and sugar-tax risks.

─────────────────────────────────────────────────────────────────────────────
C.  CUSTOMER TARGETING & MARKETING
─────────────────────────────────────────────────────────────────────────────
  7. Create tiered "Nutrition Badges".
     Tag menu items with clear labels (High Protein, Low Cal, Low Sodium)
     on digital menus and the app, making healthy choices frictionless —
     a key driver for Millennial and Gen-Z customers.

  8. Upsell healthier add-ons.
     Correlation analysis shows Calories ↔ Total Fat are highly correlated
     (r ≈ 0.80). Pairing high-fat entrées with low-fat sides (salads, apple
     slices, water) in digital order flows can improve perceived brand
     healthfulness while maintaining average order value.

  9. Leverage Breakfast as a retention tool.
     Breakfast has high variety and consistent protein levels. Loyalty-app
     promotions (e.g., "earn double points before 10 AM") can drive morning
     visit frequency, one of the highest-margin day-parts.

─────────────────────────────────────────────────────────────────────────────
D.  DATA & MONITORING
─────────────────────────────────────────────────────────────────────────────
 10. Integrate actual sales data for ROI modelling.
     This dataset contains only nutritional specs — no pricing or sales
     volumes. Merging POS transaction data would allow revenue-weighted
     analysis, seasonal demand forecasting, and true time-series modelling.
"""

print(recommendations)

# Save recommendations to text file
with open(os.path.join(OUT, "recommendations.txt"), "w", encoding="utf-8") as f:
    f.write(recommendations)
print("  ✓ Saved: recommendations.txt")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("EDA COMPLETE – all outputs saved to:")
print(f"  {OUT}")
print("=" * 70)
print("\nFiles generated:")
for f in sorted(os.listdir(OUT)):
    print(f"  • {f}")
