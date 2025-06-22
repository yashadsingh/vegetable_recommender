import os
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib
import streamlit as st

# --- Compute Dynamic Seasonality Score ---
def compute_seasonality_score(vdf):
    month_counts = vdf['Date'].dt.month.value_counts(normalize=True)
    current_month = pd.Timestamp.today().month
    return month_counts.get(current_month, 0.0)

# --- Compute Dynamic Health Score Placeholder ---
def compute_health_score(veg):
    # Placeholder for future real nutrient-based score
    return 0.5

# --- Training Function ---
def train_lstm_models():
    st.info("Retraining models. Please wait...")
    purchases_url = "https://docs.google.com/spreadsheets/d/1wdKR-b_kC79OQHFl3uat4fW6zfsjIS3Mn0M6x0J6Pgw/gviz/tq?tqx=out:csv&sheet=family_purchases"
    prefs_url = "https://docs.google.com/spreadsheets/d/1wdKR-b_kC79OQHFl3uat4fW6zfsjIS3Mn0M6x0J6Pgw/gviz/tq?tqx=out:csv&sheet=user_preferences"
    df = pd.read_csv(purchases_url)
    pref_df = pd.read_csv(prefs_url)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Year'] = df['Date'].dt.year

    sequence_length = 4
    os.makedirs("models", exist_ok=True)

    for veg in df['Vegetable'].unique():
        vdf = df[df['Vegetable'] == veg].copy()
        vdf = vdf.sort_values("Date")
        vdf['WeekIndex'] = (vdf['Year'] - vdf['Year'].min()) * 52 + vdf['Week']
        vdf = vdf.groupby('WeekIndex').agg({
            'Quantity(kg)': 'sum',
            'Leftover(kg)': 'mean'
        }).reset_index()
        vdf['WeekNum'] = vdf['WeekIndex'] % 52

        preference = pref_df[pref_df['Vegetable'] == veg]['Preference'].mean() if veg in pref_df['Vegetable'].values else 0.5
        season_score = compute_seasonality_score(df[df['Vegetable'] == veg])
        health_score = compute_health_score(veg)

        vdf['Preference'] = preference
        vdf['Seasonality'] = season_score
        vdf['Health'] = health_score

        features = vdf[['Quantity(kg)', 'Leftover(kg)', 'WeekNum', 'Preference', 'Seasonality', 'Health']].values
        if len(features) <= sequence_length:
            continue

        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)

        X, y = [], []
        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i])
            y.append(features_scaled[i][0])

        X, y = np.array(X), np.array(y)

        model = Sequential()
        model.add(LSTM(32, activation='relu', input_shape=(sequence_length, X.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=20, verbose=0)

        model_path = f"models/model_{veg.replace(' ', '_')}.h5"
        scaler_path = f"models/scaler_{veg.replace(' ', '_')}.pkl"
        model.save(model_path)
        joblib.dump(scaler, scaler_path)

    st.success("Model retraining complete.")

# --- Load LSTM Predictions ---
def get_lstm_predictions():
    purchases_url = "https://docs.google.com/spreadsheets/d/1wdKR-b_kC79OQHFl3uat4fW6zfsjIS3Mn0M6x0J6Pgw/gviz/tq?tqx=out:csv&sheet=family_purchases"
    season_url = "https://docs.google.com/spreadsheets/d/1wdKR-b_kC79OQHFl3uat4fW6zfsjIS3Mn0M6x0J6Pgw/gviz/tq?tqx=out:csv&sheet=seasonality_data"
    df = pd.read_csv(purchases_url)
    season_df = pd.read_csv(season_url)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Year'] = df['Date'].dt.year

    sequence_length = 4
    predictions = {}

    for veg in df['Vegetable'].unique():
        model_path = f"models/model_{veg.replace(' ', '_')}.h5"
        scaler_path = f"models/scaler_{veg.replace(' ', '_')}.pkl"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            continue

        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        vdf = df[df['Vegetable'] == veg].copy()
        vdf = vdf.sort_values("Date")
        vdf['WeekIndex'] = (vdf['Year'] - vdf['Year'].min()) * 52 + vdf['Week']
        vdf = vdf.groupby('WeekIndex').agg({
            'Quantity(kg)': 'sum',
            'Leftover(kg)': 'mean'
        }).reset_index()
        vdf['WeekNum'] = vdf['WeekIndex'] % 52

        preference = 0.5
        season_score = compute_seasonality_score(df[df['Vegetable'] == veg])
        health_score = compute_health_score(veg)

        vdf['Preference'] = preference
        vdf['Seasonality'] = season_score
        vdf['Health'] = health_score

        recent = vdf.tail(sequence_length)[['Quantity(kg)', 'Leftover(kg)', 'WeekNum', 'Preference', 'Seasonality', 'Health']].values
        if len(recent) < sequence_length:
            continue

        recent_scaled = scaler.transform(recent)
        input_seq = recent_scaled.reshape((1, sequence_length, recent.shape[1]))

        pred_scaled = model.predict(input_seq, verbose=0)
        inverse = scaler.inverse_transform(
            np.concatenate([pred_scaled, np.zeros((1, recent.shape[1]-1))], axis=1)
        )

        predicted_qty = round(float(inverse[0][0]), 2)
        predictions[veg] = predicted_qty

    result_df = pd.DataFrame(predictions.items(), columns=["Vegetable", "Predicted Quantity (kg)"])
    type_df = pd.read_csv(season_url)[["Vegetable", "Type"]].drop_duplicates()
    result_df = result_df.merge(type_df, on="Vegetable", how="left")
    result_df = result_df.sort_values(["Type", "Predicted Quantity (kg)"], ascending=[True, False])
    return result_df

# --- Streamlit UI ---
if __name__ == "__main__":
    st.set_page_config(page_title="Vegetable Recommender", layout="wide")
    st.title("🥦 Family Food Recommendation System")

    st.markdown("LSTM predicts quantity (kg) to buy for the coming week using preferences, recency, seasonality & health signals.")

    if st.button("🔄 Retrain LSTM Models"):
        train_lstm_models()

    pred_df = get_lstm_predictions()
    st.dataframe(pred_df, use_container_width=True)
