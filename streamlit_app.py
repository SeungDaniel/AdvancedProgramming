# streamlit_app.py
import streamlit as st
import pandas as pd
import itertools
from sklearn.ensemble import RandomForestRegressor

# ===========================================
# 🎨 Page Configuration
# ===========================================
st.set_page_config(
    page_title="Green Tea Extraction Optimizer",
    page_icon="🍵",
    layout="wide"
)

st.title("🍵 Green Tea Extraction Optimizer")
st.caption("Machine Learning–Based Optimal Brewing Condition Recommender")

# ===========================================
# 📂 Load and Preprocess Data
# ===========================================
df = pd.read_csv("green_tea_full_data_final.csv")

df.columns = (
    df.columns.str.strip()
              .str.replace(" ", "_")
              .str.replace(r"[\(\)/%]", "", regex=True)
)

df["Water_type"] = df["Water_type"].map({"Tap": 0, "Distilled": 1})
df["Tea_form"] = df["Tea_form"].map({"Leaves": 0, "Powder": 1})

# ===========================================
# 🧩 Define Ingredient Groups
# ===========================================
groups = {
    "🧪 Polyphenols": ["Flavan_3_ols", "Flavonols", "Phenolic_acid_derivatives", "Total"],
    "🧲 Minerals": ["Ca_", "Na_", "K_", "Mg_", "Mn_", "Cu_", "Fe_", "Zn_"],
    "💊 Vitamins": ["Vitamin_C", "Vitamin_B1", "Vitamin_B2", "Vitamin_B3", "Vitamin_B6"]
}

# ===========================================
# 🧭 Tab UI Setup
# ===========================================
tab_poly, tab_mineral, tab_vitamin = st.tabs(groups.keys())

# Define common logic as a function
def run_model_ui(target_list, group_name):
    st.header(group_name)
    target = st.selectbox(f"🎯 Select a target compound in {group_name}",
                          target_list, key=f"target_{group_name}")
    col1, col2 = st.columns(2)
    with col1:
        fixed_water = st.radio("💧 Water type", ["Tap", "Distilled"],
                               horizontal=True, key=f"water_{group_name}")
    with col2:
        fixed_form = st.radio("🌿 Tea form", ["Leaves", "Powder"],
                              horizontal=True, key=f"form_{group_name}")

    features = ["Water_type", "Tea_form", "Temperature", "Time_min"]
    X = df[features]
    y = df[target]

    model = RandomForestRegressor(n_estimators=400, random_state=42)
    model.fit(X, y)

    search_space = {
        "Temperature": [20, 40, 60, 80],
        "Time_min": [10, 30, 60, 180, 360]
    }

    grid = list(itertools.product(
        [1 if fixed_water == "Distilled" else 0],
        [1 if fixed_form == "Powder" else 0],
        search_space["Temperature"],
        search_space["Time_min"]
    ))
    grid_df = pd.DataFrame(grid, columns=features)
    grid_df["Predicted"] = model.predict(grid_df)

    best5 = grid_df.sort_values("Predicted", ascending=False).head(5).reset_index(drop=True)
    best_row = best5.iloc[0]

    st.success("✅ Recommended Optimal Brewing Recipe")
    st.markdown(f"""
    - **Water type:** {fixed_water}  
    - **Tea form:** {fixed_form}  
    - **Temperature:** {best_row['Temperature']} °C  
    - **Time:** {best_row['Time_min']} min  
    - **Predicted {target}:** {best_row['Predicted']:.3f}
    """)

    # === Top 5 Recommended Combinations ===
    st.markdown("#### 🌿 Top 5 Recommended Combinations")
    best5_display = best5.copy()
    best5_display["Water_type"] = best5_display["Water_type"].map({0: "Tap", 1: "Distilled"})
    best5_display["Tea_form"] = best5_display["Tea_form"].map({0: "Leaves", 1: "Powder"})
    st.dataframe(best5_display.style.format({
        "Predicted": "{:.3f}",
        "Temperature": "{:.0f}",
        "Time_min": "{:.0f}"
    }))

    # === 🔹 Visualization (Temperature vs Time)
    chart_basis = st.selectbox(
        "📊 Select axis basis for average predicted values",
        ["Temperature", "Time_min"],
        key=f"chart_basis_{group_name}"
    )
    st.markdown(f"#### Average Predicted Value Trend by {chart_basis}")
    avg_df = grid_df.groupby(chart_basis)["Predicted"].mean().reset_index()
    st.bar_chart(data=avg_df, x=chart_basis, y="Predicted")
    #st.bar_chart(grid_df.groupby("Temperature")["Predicted"].mean())

# ===========================================
# ⚙️ Run Each Tab
# ===========================================
with tab_poly:
    run_model_ui(groups["🧪 Polyphenols"], "🧪 Polyphenols")

with tab_mineral:
    run_model_ui(groups["🧲 Minerals"], "🧲 Minerals")

with tab_vitamin:
    run_model_ui(groups["💊 Vitamins"], "💊 Vitamins")

'''
streamlit run streamlit_app.py
'''
