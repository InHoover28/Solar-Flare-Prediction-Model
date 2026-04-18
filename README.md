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
