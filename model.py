import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

# ==========================================
# STEP 1: DATA ACQUISITION & STRUCTURAL SETUP
# ==========================================
print("--- Step 1: Loading Data ---")
df = pd.read_csv('hour.csv')
df['datetime'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')
df = df.set_index('datetime').sort_index()

# ==========================================
# STEP 2: TIME SERIES EDA (AS PER REQUIREMENTS)
# ==========================================
print("--- Step 2: Performing EDA ---")

# 1. Plot rental demand over time (Identifies long-term trends)
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['cnt'], color='tab:blue', linewidth=0.5)
plt.title('Requirement 1: Rental Demand Over Time (Visible Growth & Yearly Trends)')
plt.ylabel('Total Rentals (cnt)')
plt.savefig('01_demand_over_time.png')

# 2. Hourly & Daily Patterns
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
sns.boxplot(data=df, x='hr', y='cnt', ax=ax1)
ax1.set_title('Requirement 2: Hourly Pattern (Commuter Peaks)')
sns.boxplot(data=df, x='weekday', y='cnt', ax=ax2)
ax2.set_title('Requirement 2: Daily Pattern (Weekday vs Weekend)')
plt.tight_layout()
plt.savefig('02_patterns.png')

# 3. Autocorrelation (Does one hour influence the next?)
plt.figure(figsize=(10, 5))
plot_acf(df['cnt'], lags=48, ax=plt.gca())
plt.title('Requirement 3: Autocorrelation Plot (Strong Lag 1 & Lag 24 influence)')
plt.savefig('03_autocorrelation.png')

# 4. Decomposition (Trend, Seasonality, Residuals)
# Using a 1-month slice (720 hours) for visual clarity
result = seasonal_decompose(df['cnt'].head(720), model='additive', period=24)
result.plot()
plt.suptitle('Requirement 5: Time Series Decomposition', y=1.02)
plt.savefig('04_decomposition.png')

# ==========================================
# STEP 3: FEATURE ENGINEERING (THE CONTEXT NODE)
# ==========================================
print("--- Step 3: Feature Engineering ---")

def engineer_features(data):
    data = data.copy()
    # Time-based features
    data['hour_feat'] = data.index.hour
    data['dayofweek'] = data.index.dayofweek
    data['month_feat'] = data.index.month
    
    # Lag features (Requirement: Create lags)
    data['lag_1'] = data['cnt'].shift(1)   # Memory of 1 hour ago
    data['lag_24'] = data['cnt'].shift(24) # Memory of same time yesterday
    
    # Rolling statistics (Requirement: Create rolling stats)
    data['rolling_mean_3'] = data['cnt'].shift(1).rolling(window=3).mean()
    
    return data.dropna()

df_model = engineer_features(df)

# ==========================================
# STEP 4: TIME-AWARE DATA PARTITIONING
# ==========================================
print("--- Step 4: Time-Aware Split ---")
# Drop casual/registered to avoid leakage. Drop temp/atemp duplicates if desired.
features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 
            'weathersit', 'temp', 'hum', 'windspeed', 
            'hour_feat', 'dayofweek', 'month_feat', 'lag_1', 'lag_24', 'rolling_mean_3']

X = df_model[features]
y = df_model['cnt']

# Chronological split (No shuffling!)
split_idx = int(len(df_model) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ==========================================
# STEP 5: THE XGBOOST LEARNING ENGINE
# ==========================================
print("--- Step 5: Training XGBoost Model ---")
# Justification: XGBoost handles non-linear relationships and interactions efficiently.
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42, early_stopping_rounds=50)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# ==========================================
# STEP 6: EVALUATION & BASELINE COMPARISON
# ==========================================
print("--- Step 6: Evaluation & Comparison ---")
y_pred = model.predict(X_test)

# Calculate Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Baseline: Seasonal Naive (Lag 24)
y_baseline = X_test['lag_24']
baseline_mae = mean_absolute_error(y_test, y_baseline)

print(f"\nRESULTS:")
print(f"XGBoost MAE: {mae:.2f}")
print(f"XGBoost RMSE: {rmse:.2f}")
print(f"Baseline MAE: {baseline_mae:.2f}")
print(f"Improvement: {((baseline_mae - mae) / baseline_mae)*100:.2f}%")

# Plot Actual vs Predicted for the last week
plt.figure(figsize=(15, 6))
plt.plot(y_test.index[-168:], y_test.iloc[-168:], label='Actual', color='black', alpha=0.6)
plt.plot(y_test.index[-168:], y_pred[-168:], label='XGBoost Predicted', color='red', linestyle='--')
plt.title('Final Evaluation: Actual vs Predicted Demand (Last 7 Days)')
plt.legend()
plt.savefig('05_final_results.png')
print("\nSuccess! All plots saved in the current folder.")