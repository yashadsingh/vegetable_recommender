import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config("Family Veg Recommender", layout="centered")

# --- Load and Save Functions ---
@st.cache_data
def load_preferences():
    return pd.read_csv("data/user_preferences.csv")

@st.cache_data
def load_purchases():
    return pd.read_csv("data/family_purchases.csv", parse_dates=["Date"])

def save_preferences(df):
    df.to_csv("data/user_preferences.csv", index=False)

def save_purchases(df):
    df.to_csv("data/family_purchases.csv", index=False)

# --- Initialize State ---
if "pref_df" not in st.session_state:
    st.session_state.pref_df = load_preferences()

if "purchase_df" not in st.session_state:
    st.session_state.purchase_df = load_purchases()

st.title("🥦 Weekly Family Vegetable Recommender")

# --- Preferences View ---
st.subheader("👨‍👩‍👧 User Preferences (View Only)")
st.dataframe(st.session_state.pref_df)

st.subheader("➕ Add New Preference Entry")
with st.form("add_pref"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        week = st.number_input("Week", min_value=1, value=int(st.session_state.pref_df['Week'].max()) + 1 if not st.session_state.pref_df.empty else 1)
    with col2:
        user = st.selectbox("User", ["Mom", "Dad", "Kid"])
    with col3:
        veg = st.text_input("Vegetable")
    with col4:
        score = st.slider("Preference (1–5)", 1, 5, 3)

    submitted = st.form_submit_button("Add Preference")
    if submitted and veg.strip():
        new_row = pd.DataFrame([[week, user, veg, score]], columns=st.session_state.pref_df.columns)
        st.session_state.pref_df = pd.concat([st.session_state.pref_df, new_row], ignore_index=True)
        save_preferences(st.session_state.pref_df)
        st.success(f"✅ Added: {veg} rated {score} by {user}")

# --- Purchase History (Grouped by Date) ---
st.subheader("🧺 Family Purchases by Date")
grouped = st.session_state.purchase_df.groupby("Date")
for date, group in grouped:
    try:
        label = date.strftime("%Y-%m-%d")
    except Exception:
        label = str(date)
    with st.expander(f"📅 {label}", expanded=False):
        st.dataframe(group.reset_index(drop=True)[["Vegetable", "Quantity(kg)", "Leftover(kg)"]])

# --- Add Purchase ---
st.subheader("➕ Add New Family Purchase")
with st.form("add_purchase"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        date = st.date_input("Purchase Date", pd.to_datetime("today"))
    with col2:
        veg = st.text_input("Vegetable Name")
    with col3:
        qty = st.number_input("Quantity Purchased (kg)", min_value=0.1, step=0.1)
    with col4:
        left = st.number_input("Leftover (kg)", min_value=0.0, step=0.1)

    submitted = st.form_submit_button("Add Purchase")
    if submitted and veg.strip():
        new_row = pd.DataFrame([[date, veg, qty, left]], columns=["Date", "Vegetable", "Quantity(kg)", "Leftover(kg)"])
        st.session_state.purchase_df = pd.concat([st.session_state.purchase_df, new_row], ignore_index=True)
        save_purchases(st.session_state.purchase_df)
        st.success(f"✅ Added: {veg} - Qty: {qty}kg, Leftover: {left}kg")

# --- Collaborative Filtering Prediction ---
pivot_df = st.session_state.pref_df.pivot_table(index="User", columns="Vegetable", values="Preference", aggfunc="mean")
pivot_filled = pivot_df.fillna(0)
user_sim = cosine_similarity(pivot_filled)
user_sim_df = pd.DataFrame(user_sim, index=pivot_filled.index, columns=pivot_filled.index)

def predict(user, veg):
    if veg not in pivot_df.columns:
        return None
    if not np.isnan(pivot_df.get(veg, pd.Series()).get(user, np.nan)):
        return pivot_df.at[user, veg]
    sims = user_sim_df[user].drop(user)
    item_ratings = pivot_df.get(veg, pd.Series()).drop(user)
    weighted = sum(sims[i] * item_ratings[i] for i in item_ratings.index if not np.isnan(item_ratings[i]))
    sim_sum = sum(sims[i] for i in item_ratings.index if not np.isnan(item_ratings[i]))
    return round(weighted / sim_sum, 2) if sim_sum else None

# --- Build Recommendations ---
pref_vegs = set(st.session_state.pref_df["Vegetable"].dropna())
purchase_vegs = set(st.session_state.purchase_df["Vegetable"].dropna())
all_vegs = sorted(pref_vegs.union(purchase_vegs))

predictions = []

for veg in all_vegs:
    scores = []
    for user in pivot_df.index:
        score = predict(user, veg)
        if score is not None:
            scores.append(score)

    leftover_row = st.session_state.purchase_df[st.session_state.purchase_df["Vegetable"] == veg]
    avg_left = leftover_row["Leftover(kg)"].mean() if not leftover_row.empty else 0
    penalty = avg_left * 2

    if scores:
        avg_pref = np.mean(scores)
    else:
        avg_used = leftover_row["Quantity(kg)"].mean() - avg_left if not leftover_row.empty else 0
        avg_pref = 2 + (avg_used * 1.5)
        avg_pref = min(5, round(avg_pref, 2))

    final_score = avg_pref - penalty
    predictions.append((veg, round(avg_pref, 2), round(avg_left, 2), round(final_score, 2)))

# --- Show Final Recommendations ---
st.subheader("📦 Weekly Family Shopping Recommendations")
top_n = st.slider("Top N vegetables to recommend", 1, 20, 5)
df_rec = pd.DataFrame(predictions, columns=["Vegetable", "AvgPref", "AvgLeftover", "FinalScore"])
df_rec = df_rec.sort_values("FinalScore", ascending=False).head(top_n)
st.dataframe(df_rec)
