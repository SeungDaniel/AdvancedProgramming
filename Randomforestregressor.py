import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# === 1️⃣ 데이터 불러오기 ===
df = pd.read_csv("green_tea_full_data_final.csv")

# === 2️⃣ 문자형 변수 인코딩 ===
df["Water_type"] = df["Water_type"].map({"Tap": 0, "Distilled": 1})
df["Tea_form"] = df["Tea_form"].map({"Powder": 1, "Leaves": 0})

# === 3️⃣ 변수 정의 ===
features = ["Water_type", "Tea_form", "Temperature", "Time_min"]
targets = [col for col in df.columns if col not in features]

# === 4️⃣ 저장 구조 준비 ===
results = []
importances = pd.DataFrame(index=features)

# === 5️⃣ 모델 학습 루프 ===
for target in targets:
    y = df[target].values
    X = df[features].values

    model = RandomForestRegressor(
        n_estimators=400, random_state=42, n_jobs=-1
    )
    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    results.append({"Target": target, "R2": r2, "RMSE": rmse})

    # Feature importance
    importances[target] = model.feature_importances_

# === 6️⃣ 결과 DataFrame ===
results_df = pd.DataFrame(results)
print("\n📊 모델별 성능 요약")
print(results_df.sort_values("R2", ascending=False))

# === 7️⃣ 변수 중요도 수치 출력 ===
print("\n🔥 Feature Importance (수치 기반)")
print(importances.round(3))

# === 8️⃣ 각 Target별 Top Feature 해석 ===
print("\n🧠 변수별 영향력 요약")
for t in importances.columns:
    top_feat = importances[t].idxmax()
    top_val = importances[t].max()
    print(f"{t:<15} → 가장 영향 큰 변수: {top_feat} ({top_val:.2f})")

# === 9️⃣ Heatmap 시각화 ===
plt.figure(figsize=(10,6))
sns.heatmap(importances, annot=True, cmap="YlOrBr", fmt=".2f")
plt.title("Feature Importance Heatmap (RandomForest)", fontsize=13)
plt.xlabel("Target Variables")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
