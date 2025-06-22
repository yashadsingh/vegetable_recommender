
import streamlit as st
import pandas as pd
from io import BytesIO
from health_score_fetcher import get_health_score

@st.cache_data
def load_data():
    df = pd.read_csv("data/weekly_log.csv")
    df['HealthScore'] = df['Vegetable'].apply(get_health_score)
    df['Preferred'] = df['Preferred'].map({'Yes': 1, 'No': 0})
    df['Seasonal'] = df['Seasonal'].map({'Yes': 1, 'No': 0})
    df['LeftoverPenalty'] = df['Leftover(kg)'].apply(lambda x: -2 if x > 0.3 else 0)
    df['score'] = (
        (2 * df['Preferred']) +
        (1 * df['Seasonal']) +
        df['LeftoverPenalty'] +
        (2 * df['HealthScore'])
    )
    return df

df = load_data()

st.title("🥦 Weekly Vegetable Shopping AI Assistant")
st.markdown("Personalized, healthy, and budget-conscious vegetable recommendations.")

budget = st.slider("Set your weekly vegetable budget (₹)", 50, 500, 150)

df_sorted = df.sort_values(by='score', ascending=False)
total_cost = 0
recommendations = []

for _, row in df_sorted.iterrows():
    cost = row['Quantity(kg)'] * row['Price_per_kg']
    if total_cost + cost <= budget:
        recommendations.append({
            "Vegetable": row['Vegetable'],
            "Quantity (kg)": row['Quantity(kg)'],
            "Health (1-5)": row['HealthScore'],
            "Cost (₹)": f"{cost:.2f}"
        })
        total_cost += cost

st.subheader("🛒 Recommended Shopping List")
rec_df = pd.DataFrame(recommendations)
st.table(rec_df)
st.success(f"Total Cost: ₹{total_cost:.2f} / ₹{budget}")

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Recommendations')
    return output.getvalue()

excel_data = to_excel(rec_df)
st.download_button(
    label="Download as Excel",
    data=excel_data,
    file_name="vegetable_recommendations.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.subheader("📝 Add New Weekly Vegetable Entry")
with st.form("new_entry_form"):
    week = st.number_input("Week", min_value=1, value=4)
    vegetable = st.text_input("Vegetable Name")
    quantity = st.number_input("Quantity (kg)", min_value=0.1, value=1.0)
    price = st.number_input("Price per kg (₹)", min_value=1, value=20)
    seasonal = st.selectbox("Is it Seasonal?", ["Yes", "No"])
    preferred = st.selectbox("Is it Preferred?", ["Yes", "No"])
    leftover = st.number_input("Leftover from last week (kg)", min_value=0.0, value=0.0)
    submitted = st.form_submit_button("Add Entry")

    if submitted and vegetable:
        new_row = {
            "Week": week,
            "Vegetable": vegetable,
            "Quantity(kg)": quantity,
            "Price_per_kg": price,
            "Seasonal": seasonal,
            "Preferred": preferred,
            "Leftover(kg)": leftover
        }
        existing_df = pd.read_csv("data/weekly_log.csv")
        updated_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
        updated_df.to_csv("data/weekly_log.csv", index=False)
        st.success(f"✅ Added new entry for {vegetable}")
