# streamlit_app.py
import streamlit as st
import pandas as pd
import itertools
from sklearn.ensemble import RandomForestRegressor

# ===========================================
# 🎨 페이지 기본 설정
# ===========================================
st.set_page_config(
    page_title="Green Tea Extraction Optimizer",
    page_icon="🍵",
    layout="wide"
)

st.title("🍵 Green Tea Extraction Optimizer")
st.caption("Machine Learning–Based Optimal Brewing Condition Recommender")

# ===========================================
# 📂 데이터 불러오기 및 전처리
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
# 🧩 성분 그룹 정의
# ===========================================
groups = {
    "🧪 Polyphenols": ["Flavan_3_ols", "Flavonols", "Phenolic_acid_derivatives", "Total"],
    "🧲 Minerals": ["Ca_", "Na_", "K_", "Mg_", "Mn_", "Cu_", "Fe_", "Zn_"],
    "💊 Vitamins": ["Vitamin_C", "Vitamin_B1", "Vitamin_B2", "Vitamin_B3", "Vitamin_B6"]
}

# ===========================================
# 🧭 탭 UI 생성
# ===========================================
tab_poly, tab_mineral, tab_vitamin = st.tabs(groups.keys())

# 반복적으로 공통 로직 처리 함수화
def run_model_ui(target_list, group_name):
    st.header(group_name)
    target = st.selectbox(f"🎯 {group_name} 내에서 목표 성분을 선택하세요",
                          target_list, key=f"target_{group_name}")
    col1, col2 = st.columns(2)
    with col1:
        fixed_water = st.radio("💧 수질 선택", ["Tap", "Distilled"],
                               horizontal=True, key=f"water_{group_name}")
    with col2:
        fixed_form = st.radio("🌿 형태 선택", ["Leaves", "Powder"],
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

    st.success("✅ 최적 추출 레시피 제안")
    st.markdown(f"""
    - **수질:** {fixed_water}  
    - **형태:** {fixed_form}  
    - **온도:** {best_row['Temperature']} °C  
    - **시간:** {best_row['Time_min']} 분  
    - **예상 {target}:** {best_row['Predicted']:.3f}
    """)

    # === 상위 5개 조합 테이블 ===
    st.markdown("#### 🌿 상위 5개 추천 조합")
    best5_display = best5.copy()
    best5_display["Water_type"] = best5_display["Water_type"].map({0: "Tap", 1: "Distilled"})
    best5_display["Tea_form"] = best5_display["Tea_form"].map({0: "Leaves", 1: "Powder"})
    st.dataframe(best5_display.style.format({
        "Predicted": "{:.3f}",
        "Temperature": "{:.0f}",
        "Time_min": "{:.0f}"
    }))

    # === 🔹 시각화 선택 (Temperature vs Time)
    chart_basis = st.selectbox(
        "📊 어떤 축 기준으로 예측값을 볼까요?",
        ["Temperature", "Time_min"],
        key=f"chart_basis_{group_name}"
    )
    st.markdown(f"#### 평균 예측값 변화 ({chart_basis} 기준)")
    avg_df = grid_df.groupby(chart_basis)["Predicted"].mean().reset_index()
    st.bar_chart(data=avg_df, x=chart_basis, y="Predicted")
    #st.bar_chart(grid_df.groupby("Temperature")["Predicted"].mean())

    

# ===========================================
# ⚙️ 탭별 실행
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