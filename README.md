# 🌤️ Karachi Weather Predictor

A machine learning project that predicts **tomorrow's mean temperature** and **whether it will rain** in Karachi, Pakistan. Instead of calling a forecast API, I trained my own models on 11 years of historical weather data, compared several algorithms, and wrapped the best ones in an interactive Streamlit app.

**Live demo:** <<< add your Streamlit Cloud link here once deployed >>>

## What this project covers

- Collecting historical data from an API with Python
- Data cleaning, exploration and visualization (Pandas, Matplotlib, Seaborn)
- Feature engineering for time series (lags, rolling averages, cyclical seasonality)
- Regression (temperature) and classification (rain)
- Time-aware model evaluation, with no data leakage
- Comparing multiple models and explaining why one wins
- Deploying models in an interactive web app

## Dataset

- **Source:** [Open-Meteo Historical Weather API](https://open-meteo.com/) (free, no API key)
- **Location:** Karachi (24.86°N, 67.01°E)
- **Period:** 2015-01-01 to 2025-12-31 (4,018 daily rows)
- **Variables:** max, min and mean temperature, precipitation, max wind speed, mean humidity, mean surface pressure
- **Quality:** no missing values. The extreme rainfall day (304.9 mm on 2019-08-11) matches a real monsoon event, so I kept it.

Key observations from exploration:
- June is the hottest month (~30°C) and January the coolest (~19°C).
- Only about 7% of days have more than 1 mm of rain, and almost all heavy rain falls between July and September.
- Pressure has the strongest correlation with mean temperature (-0.81), largely because it follows the seasons.

## Method

**Targets**
- Regression: tomorrow's mean temperature (°C)
- Classification: will tomorrow have more than 1 mm of rain? (yes/no)

**Features (32 in total)**
- Today's weather (7 variables)
- Lag features: the same 7 variables from 1, 2 and 3 days ago
- 7-day rolling average temperature
- Day-to-day pressure change
- Seasonality encoded as sine and cosine of the day of the year, so December 31 and January 1 are treated as neighbors

**Evaluation setup**
- **Time-based split, not random:** trained on 2015-2022 (2,916 days), tested on 2023-2025 (1,095 days). A random split would let the model see days right before and after each test day, which inflates the scores.
- **Baselines:** every model must beat a simple baseline. For temperature it is "tomorrow = today". For rain it is "it never rains".
- **Time-series cross-validation** (5 folds, always training on the past and testing on the future) to check the results are not a lucky split.
- Hyperparameter tuning used only training data folds. The test set was never used to make decisions.
- Data leakage was avoided by removing the temperature target from the rain feature set.

## Results

### Temperature prediction (regression)

Test set: 2023-2025. Lower error is better.

| Model | MAE (°C) | RMSE (°C) | R² |
|---|---|---|---|
| **Linear Regression** | **0.494** | **0.686** | **0.970** |
| Gradient Boosting (tuned) | 0.497 | 0.702 | 0.968 |
| Gradient Boosting | 0.502 | 0.705 | 0.968 |
| Ridge | 0.508 | 0.707 | 0.968 |
| Random Forest | 0.524 | 0.737 | 0.965 |
| Baseline (tomorrow = today) | 0.524 | 0.761 | 0.963 |

Time-series cross-validation, mean MAE across 5 folds: Linear Regression **0.512**, Gradient Boosting 0.547, Random Forest 0.558. Linear Regression won 4 of 5 folds, with the fifth a tie.

### Rain prediction (classification)

Test set: 2023-2025, where about 9% of days are rainy.

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **Random Forest** | 0.877 | 0.372 | 0.510 | **0.430** | **0.872** |
| Logistic Regression | 0.784 | 0.259 | 0.730 | 0.382 | 0.860 |
| Baseline (never rain) | 0.909 | 0.000 | 0.000 | 0.000 | n/a |

Random Forest confusion matrix (1,094 test days):

| | Predicted dry | Predicted rain |
|---|---|---|
| **Actually dry** | 908 | 86 |
| **Actually rain** | 49 | 51 |

Lowering the decision threshold from 0.5 to 0.3 raises recall from 0.51 to 0.78 with almost no loss in precision, which would suit anyone who most wants to avoid being caught in the rain.

## Which model wins, and why?

**Temperature: Linear Regression.**
- Temperature changes slowly, so tomorrow is mostly today plus a small adjustment. That relationship is almost linear, and a straight line captures it well.
- Tree-based models cannot extrapolate beyond values seen in training, while a linear model can.
- With only about 2,900 training rows, more complex models have little extra signal to learn and can fit noise.
- Tuning pushed Gradient Boosting toward the simplest settings (shallow trees, slow learning), which suggests there is no complex pattern to find.
- The improvement over the baseline is small (0.03°C MAE), because "tomorrow = today" is already a strong baseline. The remaining ~0.5°C error is mostly unpredictable day-to-day variation.

**Rain: Random Forest.**
- Rain depends on combinations of conditions (high humidity plus falling pressure during monsoon season), which linear models handle poorly. Non-linear models do better here.
- Random Forest ranks rainy days best (highest F1 and AUC). Logistic Regression catches more rain (higher recall) but raises many more false alarms.
- The most important features were today's rainfall and pressure at several lags, so the pressure trend over recent days is the strongest signal of an incoming weather system.
- The "never rain" baseline reaches 91% accuracy while catching zero rainy days. This is why accuracy is misleading on imbalanced data and why I evaluated with precision, recall, F1 and AUC.

## Streamlit app

The app fetches the last 14 days of weather, rebuilds the same features as in training, and runs the saved models.

Features:
- Automatic prediction of tomorrow's temperature and rain score
- City picker, °C/°F toggle and adjustable rain-alert threshold
- "What-if" sliders: change today's temperature, humidity or pressure and see how the predictions react
- Recent-data explorer with multi-variable plots
- Historical explorer (daily, monthly, and seasonal views)
- Model comparison charts

## How to run

```bash
git clone https://github.com/<<< your-username >>>/weather-predictor.git
cd weather-predictor
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

To explore the analysis, open `weather.ipynb` with Jupyter.

## Project structure

```
weather-predictor/
├── weather.ipynb           # Full analysis: cleaning, EDA, features, models
├── app.py                  # Streamlit app
├── weather_raw.csv         # Historical data (Open-Meteo)
├── temp_model.pkl          # Trained Linear Regression (temperature)
├── rain_model.pkl          # Trained Random Forest (rain)
├── feature_columns.json    # Feature order expected by the models
├── requirements.txt
└── README.md
```

## Limitations

- **Trained on Karachi only.** The app lists other cities, but predictions for them are not reliable because the models never saw their climate.
- **The rain score is not a true probability.** I used balanced class weights to handle the rare rainy days, which inflates the predicted scores. It should be read as a relative signal. Proper calibration is not done yet.
- **One-day-ahead only,** using data from a single location. Real forecasts use satellite, radar and atmospheric models, so these models are not a replacement for them.
- **Rain is hard to predict from daily summaries.** An F1 of about 0.43 is respectable but far from perfect, with roughly half of rainy days missed at the default threshold.
- **Small mismatch between data sources.** Recent data in the app comes from Open-Meteo's forecast endpoint while the models were trained on its archive endpoint.
- **The what-if panel is approximate.** It changes only today's raw values, not the lag and rolling features derived from them.

## Future improvements

- Calibrate rain probabilities (for example with `CalibratedClassifierCV`)
- Choose the rain threshold using a validation set
- Train separate models per city
- Add more variables (cloud cover, dew point, wind direction) and longer lags
- Predict rainfall amount, not just rain or no rain
- Try time-series specific models (for example LSTM or SARIMAX) and compare them

## Tech stack

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Streamlit, Open-Meteo API

## Author
Aman Malik · [GitHub](https://github.com/AmanMalik2004/
