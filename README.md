# ☀️ Solar Flare Strength Predictor

> A machine learning pipeline that classifies solar flares based on strength and predicts whether an incoming solar flare will be **strong or weak** based on historical space weather patterns — built with real NASA/RHESSI observational data spanning 2008–2026.

---

## Overview

Solar flares are sudden bursts of radiation from the Sun capable of disrupting satellites, GPS systems, power grids, and communication networks on Earth. Early prediction of flare intensity is a critical problem in space weather forecasting.

This project builds an end-to-end machine learning classifier that:
- Ingests and parses **13,782 real solar flare records** from NASA/RHESSI and NOAA
- Engineers time-aware, lag-based features that only use data knowable *before* a flare peaks
- Trains a balanced **Random Forest classifier** on a chronological train/test split
- Evaluates performance with precision, recall, F1-score, and confusion matrix
- Exposes results through an **interactive Streamlit dashboard** with live single-flare prediction

---

## Dataset

| Property | Value |
|---|---|
| Source | NASA RHESSI / NOAA Space Weather |
| Time range | August 2008 → April 2026 |
| Total flare records | 13,782 |
| Features used | Duration, peak counts, total counts, timestamps, detector IDs |

Raw data is a fixed-width `.txt` file parsed into a structured Excel spreadsheet as part of the preprocessing pipeline.

---

## Approach

### 1. Data Parsing & Cleaning
- Parsed a 13,782-row fixed-width NASA text file into a structured Excel format
- Renamed and typed all columns (timestamps, numerics, string fields)
- Sorted chronologically to preserve time-series integrity

### 2. Feature Engineering
All features are constructed from **past flares only** — no data from the current flare leaks into the model:

| Feature | Description |
|---|---|
| `duration` | How long the flare has lasted at detection (seconds) |
| `hour`, `day`, `month` | Temporal position in the solar cycle |
| `gap_since_last_s` | Seconds elapsed since the previous flare |
| `rolling_peak_mean` | Average peak count of the last N flares |
| `rolling_peak_max` | Maximum peak count of the last N flares |
| `rolling_duration_mean` | Average duration of the last N flares |
| `rolling_strong_rate` | Proportion of the last N flares that were strong |

### 3. Labelling
Rather than a hardcoded threshold, flares are labelled **strong** if their peak count exceeds the **90th percentile** of the full dataset — making the threshold data-driven and adjustable.

### 4. Model
- **Algorithm:** Random Forest (200 trees, max depth 12)
- **Class balancing:** `class_weight="balanced"` to handle the natural imbalance between rare strong flares and common weak flares
- **Split:** Chronological 80/20 — no shuffling, preserving real-world temporal order
- **Reproducibility:** `random_state=42`

### 5. Evaluation
- Classification report (precision, recall, F1-score per class)
- Confusion matrix
- Feature importance ranking
- Prediction probability bar chart across the test set

---

## Results

- **Model accuracy:** 90.6%
- **Strong flare F1-score:** 17.4%
- **Top predictive feature:** `rolling_peak_mean` (past flare intensity is the strongest signal)
- **Key observation:** Class balancing significantly improves recall on rare strong flares compared to an unweighted model

---

## Interactive Dashboard

The project includes a full **Streamlit dashboard** (`app.py`) with:

- Adjustable threshold percentile, rolling window, tree count, and train/test split
- Live charts: distribution, class balance, confusion matrix, feature importance
- **Single-flare predictor** — enter pre-peak observations and get an instant strong/weak prediction with confidence score

---

## Project Structure

```
solar-flare-predictor/
│
├── data/
│   └── solar_flares.xlsx          # Parsed flare dataset (13,782 records)
│
├── Solar_Flare_Prediction_Model.py # Core ML pipeline (train, evaluate, visualise)
├── app.py                          # Streamlit interactive dashboard
├── requirements.txt                # Python dependencies
└── README.md
```

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the core model** *(terminal output + saved PNG chart)*
```bash
python Solar_Flare_Prediction_Model.py
```

**3. Launch the interactive dashboard** *(opens in browser)*
```bash
streamlit run app.py
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| pandas | Data loading and feature engineering |
| numpy | Numerical operations |
| scikit-learn | Model training and evaluation |
| matplotlib | Static visualisations |
| streamlit | Interactive dashboard |
| openpyxl | Excel file parsing |

---

## Future Improvements

- [ ] Predict **whether a flare will occur at all** (binary occurrence model)
- [ ] Incorporate GOES X-ray flux time-series for richer input features
- [ ] Experiment with LSTM/GRU networks for sequential pattern modelling
- [ ] Add solar cycle phase as a feature (sunspot number integration)
- [ ] Deploy dashboard to Streamlit Cloud for public access

---

## Author

Computer Science student with a focus on Data Science and AI, building toward aerospace and space weather applications.

---

*Data sourced from NASA's RHESSI mission and NOAA Space Weather archives.*
