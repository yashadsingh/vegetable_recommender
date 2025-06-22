import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# --- Load family_purchases.csv from the Google Sheet ---
CSV_URL = "https://docs.google.com/spreadsheets/d/1wdKR-b_kC79OQHFl3uat4fW6zfsjIS3Mn0M6x0J6Pgw/gviz/tq?tqx=out:csv&sheet=family_purchases"
df = pd.read_csv(CSV_URL)
df['Date'] = pd.to_datetime(df['Date'])
df['Week'] = df['Date'].dt.isocalendar().week
df['Year'] = df['Date'].dt.year

# --- Group by Vegetable and Week ---
sequence_length = 4
models = {}
scalers = {}

for veg in df['Vegetable'].unique():
    vdf = df[df['Vegetable'] == veg].copy()
    vdf = vdf.sort_values("Date")
    vdf['WeekIndex'] = (vdf['Year'] - vdf['Year'].min()) * 52 + vdf['Week']
    vdf = vdf.groupby('WeekIndex').agg({
        'Quantity(kg)': 'sum',
        'Leftover(kg)': 'mean'
    }).reset_index()
    vdf['WeekNum'] = vdf['WeekIndex'] % 52

    # --- Normalize ---
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(vdf[['Quantity(kg)', 'Leftover(kg)', 'WeekNum']])
    scalers[veg] = scaler

    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i])
        y.append(scaled[i][0])  # predict quantity

    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        continue

    # --- Define LSTM model ---
    model = Sequential([
        LSTM(32, input_shape=(sequence_length, 3)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)

    models[veg] = model

    # Save model and scaler
    model.save(f"model_{veg.replace(' ', '_')}.h5")
    joblib.dump(scaler, f"scaler_{veg.replace(' ', '_')}.pkl")

print("✅ Training complete. Models saved.")
