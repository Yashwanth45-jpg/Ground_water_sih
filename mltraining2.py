# aquaintel_ml_training.py

"""
AquaIntel ML Training Script – Prototype using Kaggle DWLR 2023 Dataset
Filename: DWLR_Dataset_2023.csv
Column names expected exactly:
  Date, Water_Level_m, Temperature_C, Rainfall_mm, pH, Dissolved_Oxygen_mg_L
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import datetime
from math import sqrt

print("=== AQUAINTEL TRAINING – KAGGLE DWLR PROTOTYPE ===")

# 1. Load dataset
print("Loading DWLR_Dataset_2023.csv...")
df = pd.read_csv('DWLR_Dataset_2023.csv', parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)
print(f"Records loaded: {len(df)}")

# 2. Feature engineering
print("Creating features...")
df['day_of_year'] = df['Date'].dt.dayofyear
df['month']       = df['Date'].dt.month

# Lag features
for lag in [1, 3, 7, 14, 30]:
    df[f'WL_lag_{lag}'] = df['Water_Level_m'].shift(lag)

# Rolling statistics
for w in [7, 14, 30]:
    df[f'WL_roll_mean_{w}'] = df['Water_Level_m'].rolling(w).mean()
    df[f'WL_roll_std_{w}']  = df['Water_Level_m'].rolling(w).std()

# Drop NaN rows created by shift/rolling
df = df.dropna().reset_index(drop=True)
print(f"After dropna: {len(df)} records remain")

# 3. Define features and target
features = [
    'Temperature_C', 'Rainfall_mm', 'pH', 'Dissolved_Oxygen_mg_L',
    'day_of_year', 'month',
    'WL_lag_1', 'WL_lag_3', 'WL_lag_7', 'WL_lag_14', 'WL_lag_30',
    'WL_roll_mean_7', 'WL_roll_mean_14', 'WL_roll_mean_30',
    'WL_roll_std_7', 'WL_roll_std_14', 'WL_roll_std_30'
]
X = df[features]
y = df['Water_Level_m']

# 4. Split into train and test
print("Splitting data into train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train models and select best
print("Training models...")
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'LinearRegression': LinearRegression()
}
best_model = None
best_name = None
best_r2 = -np.inf
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2  = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = sqrt(mean_squared_error(y_test, preds))
    results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
    print(f"{name}: R2={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")
    if r2 > best_r2:
        best_r2, best_name, best_model = r2, name, model

print(f"Selected best model: {best_name} (R2={best_r2:.3f})")

# 6. Crisis prediction helper
def predict_crisis(model, current_features, critical=2.0, horizon=45):
    preds = []
    # Build a DataFrame row for proper feature names
    curr_df = pd.DataFrame([current_features], columns=features)
    for day in range(horizon):
        p = model.predict(curr_df)[0]
        preds.append(p)
        # Update lag_1 for next iteration
        curr_df.at[0, 'WL_lag_1'] = p
    # Find first day crossing critical
    for i, val in enumerate(preds):
        if val <= critical:
            return i + 1, preds
    return None, preds


def format_message(days, loc="Location"):
    if days is None:
        return f"✅ {loc} water stable for next 45 days."
    if days <= 7:
        return f"🚨 URGENT: {loc} has {days} days of water left!"
    if days <= 15:
        return f"⚠️ WARNING: {loc} has {days} days left."
    return f"📊 {loc} has {days} days of water left."

# 7. Save model and metadata
print("Saving model to aquaintel_model.pkl...")
joblib.dump({
    'model': best_model,
    'features': features,
    'model_name': best_name,
    'metrics': results[best_name]
}, 'aquaintel_model.pkl')

# 8. PWA API function
def aquaintel_predict_api(loc_name, temp, rain, ph, do, current_wl):
    now = datetime.datetime.now()
    curr = {
        'Temperature_C': temp,
        'Rainfall_mm': rain,
        'pH': ph,
        'Dissolved_Oxygen_mg_L': do,
        'day_of_year': now.timetuple().tm_yday,
        'month': now.month,
        'WL_lag_1': current_wl,
        'WL_lag_3': current_wl,
        'WL_lag_7': current_wl,
        'WL_lag_14': current_wl,
        'WL_lag_30': current_wl,
        'WL_roll_mean_7': current_wl,
        'WL_roll_mean_14': current_wl,
        'WL_roll_mean_30': current_wl,
        'WL_roll_std_7': 0,
        'WL_roll_std_14': 0,
        'WL_roll_std_30': 0
    }
    days, forecast = predict_crisis(best_model, curr)
    msg = format_message(days, loc_name)
    alert = msg.split()[0].strip("🚨⚠️📊✅")
    return {
        'village_name': loc_name,
        'days_until_crisis': days,
        'crisis_message': msg,
        'alert_level': alert,
        'recommendations': [
            "Implement water conservation" if days and days <= 15 else "Continue regular monitoring"
        ],
        'forecast_7_days': forecast[:7],
        'model_type': best_name,
        'confidence': best_r2
    }

# 9. Test API
print("\nTesting API helper...")
sample_wl = y_test.iloc[0]
test_out = aquaintel_predict_api(
    "TestSite", 
    temp=X_test['Temperature_C'].iloc[0], 
    rain=X_test['Rainfall_mm'].iloc[0], 
    ph=X_test['pH'].iloc[0], 
    do=X_test['Dissolved_Oxygen_mg_L'].iloc[0], 
    current_wl=sample_wl
)
print(test_out)

print("\n✅ Training complete. Model saved.")
