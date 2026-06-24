import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

EXCEL_PATH = r"C:\Users\india\documents\solar_flares.xlsx"
RANDOM_SEED = 42
STRONG_FLARE_PERCENTILE = 75   # top 25% of peak counts = "strong"
ROLLING_WINDOW = 5             # how many past flares to look back

# =============================================================
# 1. LOAD
# =============================================================

df = pd.read_excel(EXCEL_PATH)

df = df.rename(columns={
    "Flare":        "flare_id",
    "Start time":   "start_time",
    "Dur (s)":      "duration",
    "Peak (c/s)":   "peak_counts",
    "Total Count":  "total_counts"
})

df = df[["flare_id", "start_time", "duration", "peak_counts", "total_counts"]]
df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
df = df.dropna()
df = df.sort_values("start_time").reset_index(drop=True)

print(f"Loaded {len(df):,} flares  |  {df['start_time'].min().date()} to {df['start_time'].max().date()}")

# =============================================================
# 2. LABEL  (data-driven threshold, not hardcoded)
# =============================================================

threshold = df["peak_counts"].quantile(STRONG_FLARE_PERCENTILE / 100)
df["strong_flare"] = (df["peak_counts"] > threshold).astype(int)

print(f"\nStrong-flare threshold: {threshold:,.0f} c/s  (top {100 - STRONG_FLARE_PERCENTILE}%)")
print(f"  Weak   (0): {(df['strong_flare']==0).sum():,}")
print(f"  Strong (1): {(df['strong_flare']==1).sum():,}")

# =============================================================
# 3. FEATURE ENGINEERING  (only knowable BEFORE the flare peaks)
# =============================================================

df["hour"]  = df["start_time"].dt.hour
df["day"]   = df["start_time"].dt.day
df["month"] = df["start_time"].dt.month

# How long since the last flare?
df["gap_since_last_s"] = df["start_time"].diff().dt.total_seconds().fillna(0)

# Rolling stats of PREVIOUS flares (shift(1) so we never leak the current flare's data)
df["rolling_peak_mean"]     = df["peak_counts"].shift(1).rolling(ROLLING_WINDOW).mean()
df["rolling_peak_max"]      = df["peak_counts"].shift(1).rolling(ROLLING_WINDOW).max()
df["rolling_duration_mean"] = df["duration"].shift(1).rolling(ROLLING_WINDOW).mean()
df["rolling_strong_rate"]   = df["strong_flare"].shift(1).rolling(ROLLING_WINDOW).mean()

df = df.dropna().reset_index(drop=True)
print(f"\nAfter feature engineering: {len(df):,} samples")

# =============================================================
# 4. TRAIN / TEST SPLIT  (chronological — no shuffling)
# =============================================================

FEATURES = [
    "duration",
    "hour",
    "day",
    "month",
    "gap_since_last_s",
    "rolling_peak_mean",
    "rolling_peak_max",
    "rolling_duration_mean",
    "rolling_strong_rate",
]

X = df[FEATURES]
y = df["strong_flare"]

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# =============================================================
# 5. MODEL  (class_weight balances the majority/minority split)
# =============================================================

classes = np.array([0, 1])
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=RANDOM_SEED,
    class_weight=class_weight_dict,
    n_jobs=-1
)
model.fit(X_train, y_train)

# =============================================================
# 6. EVALUATION
# =============================================================

predictions   = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\n" + "="*55)
print("MODEL EVALUATION")
print("="*55)
print(classification_report(y_test, predictions, target_names=["Weak", "Strong"]))

# =============================================================
# 7. VISUALISATIONS
# =============================================================

fig = plt.figure(figsize=(14, 10))
fig.suptitle("Solar Flare Strength Classifier — Results", fontsize=14, fontweight="bold", y=0.98)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(df["peak_counts"], bins=60, color="#378ADD", alpha=0.7, label="All flares")
ax1.axvline(threshold, color="#E24B4A", linewidth=2, linestyle="--",
            label=f"Threshold ({threshold:,.0f} c/s)")
ax1.set_title("Peak Count Distribution")
ax1.set_xlabel("Peak Counts (c/s)")
ax1.set_ylabel("Frequency")
ax1.legend()

ax2 = fig.add_subplot(gs[0, 2])
counts = df["strong_flare"].value_counts()
ax2.pie(counts, labels=["Weak", "Strong"], colors=["#378ADD", "#E24B4A"],
        autopct="%1.1f%%", startangle=90)
ax2.set_title("Class Balance")

ax3 = fig.add_subplot(gs[1, 0])
cm   = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Weak", "Strong"])
disp.plot(ax=ax3, colorbar=False, cmap="Blues")
ax3.set_title("Confusion Matrix")

ax4 = fig.add_subplot(gs[1, 1:])
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
colors = ["#E24B4A" if v == importances.max() else "#378ADD" for v in importances]
importances.plot(kind="barh", ax=ax4, color=colors)
ax4.set_title("Feature Importance")
ax4.set_xlabel("Importance")

plt.savefig("solar_flare_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to solar_flare_results.png")
