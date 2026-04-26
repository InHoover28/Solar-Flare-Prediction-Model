import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

# =============================================================
# PAGE CONFIG
# =============================================================

st.set_page_config(
    page_title="Solar Flare Predictor",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ Solar Flare Strength Classifier")
st.caption("Random Forest model trained on RHESSI/GBM flare data (2008–2026)")

# =============================================================
# SIDEBAR CONTROLS
# =============================================================

st.sidebar.header("Model Settings")

excel_path = st.sidebar.text_input(
    "Excel file path",
    value=r"C:\Users\india\documents\solar_flares.xlsx"
)

percentile = st.sidebar.slider(
    "Strong flare threshold (percentile)",
    min_value=50, max_value=95, value=75, step=5,
    help="Flares above this percentile of peak counts are labelled 'strong'"
)

rolling_window = st.sidebar.slider(
    "Rolling window (# past flares)",
    min_value=3, max_value=20, value=5
)

n_estimators = st.sidebar.slider(
    "Number of trees",
    min_value=50, max_value=500, value=200, step=50
)

train_split = st.sidebar.slider(
    "Train / test split",
    min_value=0.6, max_value=0.9, value=0.8, step=0.05,
    help="Fraction of data used for training (chronological)"
)

run_btn = st.sidebar.button("Run Model", type="primary", use_container_width=True)

# =============================================================
# LOAD & CACHE
# =============================================================

@st.cache_data
def load_data(path):
    df = pd.read_excel(path)
    df = df.rename(columns={
        "Flare":       "flare_id",
        "Start time":  "start_time",
        "Dur (s)":     "duration",
        "Peak (c/s)":  "peak_counts",
        "Total Count": "total_counts"
    })
    df = df[["flare_id", "start_time", "duration", "peak_counts", "total_counts"]]
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df = df.dropna().sort_values("start_time").reset_index(drop=True)
    return df

# =============================================================
# MAIN
# =============================================================

if run_btn or "results" in st.session_state:

    with st.spinner("Loading data..."):
        try:
            df = load_data(excel_path)
        except Exception as e:
            st.error(f"Could not load file: {e}")
            st.stop()

    # --- Labels ---
    threshold = df["peak_counts"].quantile(percentile / 100)
    df["strong_flare"] = (df["peak_counts"] > threshold).astype(int)

    # --- Features ---
    df["hour"] = df["start_time"].dt.hour
    df["day"]  = df["start_time"].dt.day
    df["month"] = df["start_time"].dt.month
    df["gap_since_last_s"] = df["start_time"].diff().dt.total_seconds().fillna(0)
    df["rolling_peak_mean"]     = df["peak_counts"].shift(1).rolling(rolling_window).mean()
    df["rolling_peak_max"]      = df["peak_counts"].shift(1).rolling(rolling_window).max()
    df["rolling_duration_mean"] = df["duration"].shift(1).rolling(rolling_window).mean()
    df["rolling_strong_rate"]   = df["strong_flare"].shift(1).rolling(rolling_window).mean()
    df = df.dropna().reset_index(drop=True)

    FEATURES = [
        "duration", "hour", "day", "month", "gap_since_last_s",
        "rolling_peak_mean", "rolling_peak_max",
        "rolling_duration_mean", "rolling_strong_rate"
    ]

    X, y = df[FEATURES], df["strong_flare"]
    split_idx = int(len(df) * train_split)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    with st.spinner("Training model..."):
        classes = np.array([0, 1])
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            random_state=42,
            class_weight=dict(zip(classes, weights)),
            n_jobs=-1
        )
        model.fit(X_train, y_train)

    preds  = model.predict(X_test)
    probs  = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, preds, target_names=["Weak", "Strong"], output_dict=True)

    # =============================================================
    # METRICS ROW
    # =============================================================

    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total flares", f"{len(df):,}")
    c2.metric("Training set", f"{len(X_train):,}")
    c3.metric("Test set",     f"{len(X_test):,}")
    c4.metric("Accuracy",     f"{report['accuracy']*100:.1f}%")
    c5.metric("Threshold",    f"{threshold:,.0f} c/s")

    st.divider()

    # =============================================================
    # CHARTS ROW 1
    # =============================================================

    col_a, col_b, col_c = st.columns([2, 1, 1])

    with col_a:
        st.subheader("Peak count distribution")
        fig1, ax1 = plt.subplots(figsize=(7, 3))
        ax1.hist(df["peak_counts"], bins=60, color="#378ADD", alpha=0.75)
        ax1.axvline(threshold, color="#E24B4A", linewidth=2, linestyle="--",
                    label=f"Threshold: {threshold:,.0f} c/s")
        ax1.set_xlabel("Peak Counts (c/s)")
        ax1.set_ylabel("Frequency")
        ax1.legend(fontsize=9)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col_b:
        st.subheader("Class balance")
        fig2, ax2 = plt.subplots(figsize=(3, 3))
        counts = df["strong_flare"].value_counts()
        ax2.pie(counts, labels=["Weak", "Strong"],
                colors=["#378ADD", "#E24B4A"],
                autopct="%1.1f%%", startangle=90)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    with col_c:
        st.subheader("Confusion matrix")
        fig3, ax3 = plt.subplots(figsize=(3, 3))
        cm = confusion_matrix(y_test, preds)
        ConfusionMatrixDisplay(cm, display_labels=["Weak", "Strong"]).plot(
            ax=ax3, colorbar=False, cmap="Blues"
        )
        ax3.set_title("")
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    # =============================================================
    # CHARTS ROW 2
    # =============================================================

    col_d, col_e = st.columns([1, 1])

    with col_d:
        st.subheader("Feature importance")
        imps = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        colors = ["#E24B4A" if v == imps.max() else "#378ADD" for v in imps]
        imps.plot(kind="barh", ax=ax4, color=colors)
        ax4.set_xlabel("Importance")
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    with col_e:
        st.subheader("Classification report")
        report_df = pd.DataFrame(report).T.drop("accuracy", errors="ignore")
        report_df = report_df[["precision", "recall", "f1-score", "support"]].round(3)
        report_df["support"] = report_df["support"].astype(int, errors="ignore")
        st.dataframe(report_df, use_container_width=True)

        st.subheader("Prediction probability — test set (first 200)")
        fig5, ax5 = plt.subplots(figsize=(5, 2.5))
        test_slice = probs[:200]
        colors_p = ["#E24B4A" if p > 0.5 else "#378ADD" for p in test_slice]
        ax5.bar(range(len(test_slice)), test_slice, color=colors_p, width=1.0)
        ax5.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
        ax5.set_ylabel("P(strong)")
        ax5.set_xlabel("Test sample index")
        fig5.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

    # =============================================================
    # PREDICT A SINGLE FLARE
    # =============================================================

    st.divider()
    st.subheader("🔭 Predict a single flare")
    st.caption("Enter what you know *before* the flare peaks to get a prediction.")

    p1, p2, p3 = st.columns(3)
    inp_dur      = p1.number_input("Duration so far (s)",        min_value=0,   value=300)
    inp_hour     = p1.number_input("Hour of day (0–23)",         min_value=0,   max_value=23, value=12)
    inp_month    = p2.number_input("Month (1–12)",               min_value=1,   max_value=12, value=6)
    inp_gap      = p2.number_input("Seconds since last flare",   min_value=0,   value=3600)
    inp_rp_mean  = p3.number_input("Avg peak of last 5 flares (c/s)", min_value=0.0, value=5000.0)
    inp_rp_max   = p3.number_input("Max peak of last 5 flares (c/s)", min_value=0.0, value=8000.0)
    inp_rd_mean  = p3.number_input("Avg duration of last 5 flares (s)", min_value=0.0, value=400.0)
    inp_rs_rate  = p2.number_input("Strong-flare rate of last 5 (0–1)", min_value=0.0, max_value=1.0, value=0.4)

    if st.button("Predict", type="primary"):
        sample = pd.DataFrame([{
            "duration":              inp_dur,
            "hour":                  inp_hour,
            "day":                   15,
            "month":                 inp_month,
            "gap_since_last_s":      inp_gap,
            "rolling_peak_mean":     inp_rp_mean,
            "rolling_peak_max":      inp_rp_max,
            "rolling_duration_mean": inp_rd_mean,
            "rolling_strong_rate":   inp_rs_rate,
        }])
        prob = model.predict_proba(sample)[0][1]
        label = "Strong" if prob > 0.5 else "Weak"
        color = "red" if label == "Strong" else "blue"
        st.markdown(f"### Prediction: :{color}[{label} flare]")
        st.progress(float(prob), text=f"Confidence: {prob*100:.1f}% probability of strong flare")

else:
    st.info("Configure the settings in the sidebar and click **Run Model** to start.")
    st.markdown("""
    **What this model does:**
    - Loads your `solar_flares.xlsx` dataset
    - Labels each flare as *strong* or *weak* based on a data-driven peak-count threshold
    - Engineers features from *past* flares only (no data leakage)
    - Trains a balanced Random Forest on 80% of the data chronologically
    - Evaluates on the remaining 20% and lets you predict individual flares

    **To run:** Make sure your Excel file is at the path shown in the sidebar, then click **Run Model**.
    """)
