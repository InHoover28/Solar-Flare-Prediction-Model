# Solar-Flare-Prediction-Using-Machine-Learning

OVERVIEW

Solar flares are sudden bursts of radiation from the Sun that can disrupt satellites, GPS systems, and communication networks from on Earth. 
This project uses real-world space weather data from NASA and NOAA to build a machine learning model capable of predicting solar flare activity based on historical patterns. 
The goal is to simulate a real-world aerospace/data science workflow: ingesting raw data, engineering features, training models, and evaluating predictive performance. 

DATA SOURCES

  -NASA Open Data Portal
  -NOAA Space Weather Data
These datasets include time-series measurements of solar activity such as solar flux, magentic field data, and flare event records.

APPROACH

  1. Data Processing
    -Cleaned missing and inconsistent values
    -Converted timestamps into structured datetime formats
    -Sorted fata chronologically for time-series integrity

  2. Featured Engineering
    -Extracted temporal features (hour, day, etc.)
    -Created rolling averages to capture trends
    -Generated lag-based features to model temporal dependencies

  3. Model Development
    -Implemented baseline models (Logistic Regression, Random Forest)
    -Trained on historical data to predict flare occurence
    -Preserved time order during train/test split

  4. Evaluation
    -Classification metrics (precision, recall, F1-score)
    -Comparison of predicted vs actual flare events
    -Analysis of model limitations and performance tradeoffs

RESULTS

UPDATE THIS SECTION ONCE MODEL IS COMPLETE
--Model Accuracy: XX%
--Key Insight:(example:solar flux trends strongly correlate with flare likelihood)
--Observations:(example: model struggles with rare high-intensity flare events)

TECH STACK

-Python
-pandas
-numpy
-scikit-learn
-matplotlib

PROJECT STRUCTURE

solar-flare-predictor/ 
│ 
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── exploration.ipynb 
│ 
├── src/ 
│   ├── data_loader.py 
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py 
│   ├── evaluate.py
│   
├── main.py
├── requirements.txt 
└── README.md

HOW TO RUN 

pip install -r requirements.txt
python main.py

WHY THIS PROJECT MATTERS

Solar flare prediction plays a critical role in protecting:
-Satellites and spacecraft
-Communication systems
-Power grids on Earth
This project demonstrates how machine learning can be applied to real aerospace and space weather problems using publically available data.

FUTURE IMPROVEMENTS

-Implement deep learning models (LSTM for time-series prediction)
-Incorporate additional space weather variables
-Improve handling of imbalanced data (rare flare events)
-Deploy as an interactive dashboard of API

AUTHOR
Computer Science student with a focus on Data Science and AI, interested in Aerospace applications and real-world machine learning systems.
