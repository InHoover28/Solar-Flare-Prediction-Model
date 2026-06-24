import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

st.set_page_config(page_title="Solar Flare Predictor", page_icon="☀️", layout="wide")
st.title("☀️ Solar Flare Strength Classifier")
st.caption("Random Forest model trained on RHESSI/GBM flare data (2008–2026)")

# =============================================================
# SIDEBAR
# =============================================================

st.sidebar.header("Model Settings")
excel_path     = st.sidebar.text_input("Excel file path", value="solar_flares.xlsx")
percentile     = st.sidebar.slider("Strong flare threshold (percentile)", 50, 95, 90, 5)
rolling_window = st.sidebar.slider("Rolling window (# past flares)", 3, 20, 5)
n_estimators   = st.sidebar.slider("Number of trees", 50, 500, 200, 50)
train_split    = st.sidebar.slider("Train / test split", 0.6, 0.9, 0.8, 0.05)
run_btn        = st.sidebar.button("Run Model", type="primary", use_container_width=True)

# =============================================================
# LOAD
# =============================================================

@st.cache_data
def load_data(path):
    df = pd.read_excel(path)
    df = df.rename(columns={
        "Flare": "flare_id", "Start time": "start_time",
        "Dur (s)": "duration", "Peak (c/s)": "peak_counts", "Total Count": "total_counts"
    })
    df = df[["flare_id","start_time","duration","peak_counts","total_counts"]]
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    return df.dropna().sort_values("start_time").reset_index(drop=True)

# =============================================================
# TRAIN
# =============================================================

@st.cache_resource
def train_model(path, percentile, rolling_window, n_estimators, train_split):
    df = load_data(path)
    threshold = df["peak_counts"].quantile(percentile / 100)
    df["strong_flare"] = (df["peak_counts"] > threshold).astype(int)
    df["hour"]  = df["start_time"].dt.hour
    df["day"]   = df["start_time"].dt.day
    df["month"] = df["start_time"].dt.month
    df["gap_since_last_s"]      = df["start_time"].diff().dt.total_seconds().fillna(0)
    df["rolling_peak_mean"]     = df["peak_counts"].shift(1).rolling(rolling_window).mean()
    df["rolling_peak_max"]      = df["peak_counts"].shift(1).rolling(rolling_window).max()
    df["rolling_duration_mean"] = df["duration"].shift(1).rolling(rolling_window).mean()
    df["rolling_strong_rate"]   = df["strong_flare"].shift(1).rolling(rolling_window).mean()
    df = df.dropna().reset_index(drop=True)
    FEATURES = ["duration","peak_counts","hour","day","month","gap_since_last_s",
                "rolling_peak_mean","rolling_peak_max","rolling_duration_mean","rolling_strong_rate"]
    X, y = df[FEATURES], df["strong_flare"]
    split_idx = int(len(df) * train_split)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=12,
                                   random_state=42, class_weight=dict(zip(classes, weights)), n_jobs=-1)
    model.fit(X_train, y_train)
    preds  = model.predict(X_test)
    probs  = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, preds, target_names=["Weak","Strong"], output_dict=True)
    return model, df, threshold, X_train, X_test, y_test, preds, probs, report, FEATURES

# =============================================================
# LANDING STATE — before Run Model is clicked
# =============================================================

if run_btn:
    st.session_state["model_params"] = (excel_path, percentile, rolling_window, n_estimators, train_split)

if "model_params" not in st.session_state:
    st.info("Configure the settings in the sidebar and click **Run Model** to start.")
    st.markdown("""
    **What this model does:**
    - Loads your `solar_flares.xlsx` dataset
    - Labels each flare as *strong* or *weak* based on a data-driven peak-count threshold
    - Engineers features from *past* flares only (no data leakage)
    - Trains a balanced Random Forest on 80% of the data chronologically
    - Evaluates on the remaining 20% and lets you predict individual flares
    """)

else:
    # =============================================================
    # EVERYTHING below only runs after Run Model is clicked
    # =============================================================

    with st.spinner("Training model..."):
        try:
            model, df, threshold, X_train, X_test, y_test, preds, probs, report, FEATURES = train_model(*st.session_state["model_params"])
        except Exception as e:
            st.error(f"Could not load file or train model: {e}")
            st.stop()

    weak_f1       = report["Weak"]["f1-score"]
    strong_f1     = report["Strong"]["f1-score"]
    strong_recall = report["Strong"]["recall"]

    # --- METRICS ---
    st.divider()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total flares",    f"{len(df):,}")
    c2.metric("Training set",    f"{len(X_train):,}")
    c3.metric("Test set",        f"{len(X_test):,}")
    c4.metric("Accuracy",        f"{report['accuracy']*100:.1f}%")
    c5.metric("Strong flare F1", f"{strong_f1*100:.1f}%")
    c6.metric("Strong recall",   f"{strong_recall*100:.1f}%")
    st.divider()

    # --- CHART 1: Predictions vs Reality ---
    st.subheader("Model Predictions vs Reality")
    st.caption("Every flare in the test set — did the model get it right?")

    test_df = df.iloc[len(X_train):].copy().reset_index(drop=True)
    test_df["predicted"] = preds
    test_df["actual"]    = y_test.values
    test_df["correct"]   = (test_df["predicted"] == test_df["actual"]).astype(int)
    test_df["label"] = test_df.apply(
        lambda r: "Correct — Strong" if r.actual==1 and r.correct==1
        else ("Correct — Weak"  if r.actual==0 and r.correct==1
        else ("Missed Strong"   if r.actual==1 and r.correct==0
        else  "False Alarm")), axis=1)

    color_map = {"Correct — Strong":"#E24B4A","Correct — Weak":"#378ADD",
                 "Missed Strong":"#FF8C00","False Alarm":"#9B59B6"}

    fig1, ax1 = plt.subplots(figsize=(14, 4))
    weak_mask   = test_df["actual"] == 0
    caught_mask = test_df["label"] == "Correct — Strong"
    missed_mask = test_df["label"] == "Missed Strong"
    false_mask  = test_df["label"] == "False Alarm"
    ax1.scatter(test_df[weak_mask].index, test_df[weak_mask]["peak_counts"],
                c="#CCCCCC", s=6, alpha=0.4, label="Weak flare", zorder=1)
    ax1.scatter(test_df[caught_mask].index, test_df[caught_mask]["peak_counts"],
                c="#E24B4A", s=30, alpha=0.9, label=f"Correctly caught strong ({caught_mask.sum()})", zorder=3)
    if missed_mask.sum() > 0:
        ax1.scatter(test_df[missed_mask].index, test_df[missed_mask]["peak_counts"],
                    c="#FF8C00", s=50, marker="x", linewidths=2,
                    label=f"Missed strong ({missed_mask.sum()})", zorder=4)
    if false_mask.sum() > 0:
        ax1.scatter(test_df[false_mask].index, test_df[false_mask]["peak_counts"],
                    c="#9B59B6", s=30, marker="^",
                    label=f"False alarm ({false_mask.sum()})", zorder=4)
    ax1.axhline(threshold, color="black", linewidth=1.5, linestyle="--",
                label=f"Strong threshold ({threshold:,.0f} c/s)", zorder=2)
    ax1.set_xlabel("Flare index (chronological)")
    ax1.set_ylabel("Peak count (c/s)")
    ax1.set_yscale("log")
    ax1.legend(fontsize=8, loc="upper left", ncol=2)
    ax1.set_title("Strong flares stand out clearly above the threshold — red = correctly caught")
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    st.divider()

    # --- CHART 2: Breakdown table + Confidence bars ---
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.subheader("Classification breakdown")
        st.caption("How the model's predictions line up with reality")
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        total = len(y_test)
        breakdown_df = pd.DataFrame({
            "Result":    ["Correct — Weak","Correct — Strong","Missed Strong (critical)","False Alarm"],
            "Count":     [tn, tp, fn, fp],
            "% of test": [f"{tn/total*100:.1f}%", f"{tp/total*100:.1f}%",
                          f"{fn/total*100:.1f}%", f"{fp/total*100:.1f}%"],
        })
        def highlight_row(row):
            if "Missed" in row["Result"]:
                return ["background-color:#fff0e0;color:#cc5500"]*len(row)
            elif "Correct" in row["Result"]:
                return ["background-color:#e8f5e9;color:#2e7d32"]*len(row)
            else:
                return ["background-color:#f3e5f5;color:#6a1b9a"]*len(row)
        st.dataframe(breakdown_df.style.apply(highlight_row, axis=1),
                     use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("How confident was the model on each prediction?")
        st.caption("Red = predicted strong  |  Blue = predicted weak  |  Orange = missed strong flare")
        display_n    = min(300, len(probs))
        slice_probs  = probs[:display_n]
        slice_preds  = preds[:display_n]
        slice_actual = y_test.values[:display_n]
        bar_colors   = []
        for i in range(display_n):
            if slice_actual[i]==1 and slice_preds[i]==0:
                bar_colors.append("#FF8C00")
            elif slice_preds[i]==1:
                bar_colors.append("#E24B4A")
            else:
                bar_colors.append("#378ADD")
        fig2, ax2 = plt.subplots(figsize=(10, 3.5))
        ax2.bar(range(display_n), slice_probs, color=bar_colors, width=1.0, alpha=0.85)
        ax2.axhline(0.5, color="black", linewidth=1.2, linestyle="--")
        ax2.set_ylabel("P(strong flare)")
        ax2.set_xlabel("Test flares (chronological order)")
        ax2.set_ylim(0, 1)
        strong_patch = mpatches.Patch(color="#E24B4A", label="Predicted strong")
        weak_patch   = mpatches.Patch(color="#378ADD", label="Predicted weak")
        missed_patch = mpatches.Patch(color="#FF8C00", label="Missed strong flare")
        ax2.legend(handles=[strong_patch, weak_patch, missed_patch], fontsize=8, loc="upper right")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.divider()

    # --- CHART 3: Feature importance + Strong flare timeline ---
    col_c, col_d = st.columns([1, 2])

    with col_c:
        st.subheader("What drives the prediction?")
        readable = {
            "peak_counts":"Current peak count","rolling_peak_mean":"Avg peak — last 5 flares",
            "rolling_peak_max":"Max peak — last 5 flares","rolling_duration_mean":"Avg duration — last 5",
            "rolling_strong_rate":"Strong rate — last 5","gap_since_last_s":"Time since last flare",
            "duration":"Duration so far","month":"Month","hour":"Hour of day","day":"Day of month"
        }
        imps = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
        imps.index = [readable.get(i, i) for i in imps.index]
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        bar_colors = ["#E24B4A" if v == imps.max() else "#378ADD" for v in imps]
        imps.plot(kind="barh", ax=ax3, color=bar_colors)
        ax3.set_xlabel("Importance score")
        for i, val in enumerate(imps):
            ax3.text(val + 0.002, i, f"{val*100:.1f}%", va="center", fontsize=8)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with col_d:
        st.subheader("Strong flare detections over time")
        st.caption("When did strong flares occur — and did the model catch them?")
        strong_actual    = test_df[test_df["actual"] == 1]
        correctly_caught = strong_actual[strong_actual["correct"] == 1]
        missed           = strong_actual[strong_actual["correct"] == 0]
        fig4, ax4 = plt.subplots(figsize=(10, 3.5))
        ax4.scatter(correctly_caught["start_time"], correctly_caught["peak_counts"],
                    c="#E24B4A", s=25, alpha=0.8, label=f"Correctly caught ({len(correctly_caught)})", zorder=3)
        ax4.scatter(missed["start_time"], missed["peak_counts"],
                    c="#FF8C00", s=35, marker="x", linewidths=1.5,
                    label=f"Missed ({len(missed)})", zorder=4)
        ax4.axhline(threshold, color="black", linewidth=1, linestyle="--",
                    label=f"Threshold ({threshold:,.0f} c/s)")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Peak count (c/s)")
        ax4.legend(fontsize=8)
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    st.divider()

    # =============================================================
    # PREDICT A SINGLE FLARE
    # =============================================================

    st.subheader("Predict a single flare")
    st.caption("Enter current flare observations and recent history to get a prediction.")

    p1, p2, p3 = st.columns(3)
    inp_dur     = p1.number_input("Duration so far (s)",                min_value=0,   value=300)
    inp_peak    = p1.number_input("Current peak count (c/s)",           min_value=0.0, value=3000.0)
    inp_hour    = p1.number_input("Hour of day (0-23)",                 min_value=0,   max_value=23, value=12)
    inp_month   = p2.number_input("Month (1-12)",                       min_value=1,   max_value=12, value=6)
    inp_gap     = p2.number_input("Seconds since last flare",           min_value=0,   value=3600)
    inp_rs_rate = p2.number_input("Strong-flare rate of last 5 (0-1)", min_value=0.0, max_value=1.0, value=0.4)
    inp_rp_mean = p3.number_input("Avg peak of last 5 flares (c/s)",   min_value=0.0, value=5000.0)
    inp_rp_max  = p3.number_input("Max peak of last 5 flares (c/s)",   min_value=0.0, value=8000.0)
    inp_rd_mean = p3.number_input("Avg duration of last 5 flares (s)", min_value=0.0, value=400.0)

    if st.button("Predict", type="primary"):
        sample = pd.DataFrame([{
            "duration": inp_dur, "peak_counts": inp_peak, "hour": inp_hour,
            "day": 15, "month": inp_month, "gap_since_last_s": inp_gap,
            "rolling_peak_mean": inp_rp_mean, "rolling_peak_max": inp_rp_max,
            "rolling_duration_mean": inp_rd_mean, "rolling_strong_rate": inp_rs_rate,
        }])
        prob  = model.predict_proba(sample)[0][1]
        label = "Strong" if prob > 0.5 else "Weak"
        color = "red" if label == "Strong" else "blue"
        st.markdown(f"### Prediction: :{color}[{label} flare]")
        st.progress(float(prob), text=f"Confidence: {prob*100:.1f}% probability of strong flare")
