import pandas as pd
import re

# === 1️⃣ 데이터 불러오기 ===
df = pd.read_csv("green_tea_full_data.csv", encoding="utf-8-sig")

# === 2️⃣ 컬럼명 정리 (공백, 특수문자 제거) ===
df.columns = (
    df.columns
    .str.replace(r"\(.*\)", "", regex=True)      # 단위 괄호 제거
    .str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)  # 특수문자 → _
    .str.strip()
)

# === 3️⃣ ± 이후 값 제거 (평균값만 남기기) ===
for col in df.columns:
    if col not in ["Water_type", "Tea_form", "TempTime"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"±.*", "", regex=True)  # ± 뒤 오차 제거
            .str.replace(r"[^\d\.]", "", regex=True)  # 숫자와 점만 남김
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

# === 4️⃣ TempTime 분리 (Temperature, Time_min) ===
def parse_temp_time(val):
    """TempTime 예: '85 °C/10 min' → (85, 10)"""
    if pd.isna(val):
        return None, None
    temp_match = re.search(r"(\d+)", str(val))
    time_match = re.search(r"/(\d+)\s*(min|h)", str(val))
    if not temp_match or not time_match:
        return None, None
    temp = int(temp_match.group(1))
    time = int(time_match.group(1))
    if "h" in time_match.group(2):  # 시간을 분으로 변환
        time *= 60
    return temp, time

df["Temperature"], df["Time_min"] = zip(*df["TempTime"].apply(parse_temp_time))

# === 5️⃣ 단위 변환 (모두 mg/L로 통일) ===
# mg/L : Ca, Na, K, Mg, Mn (변환 X)
# µg/L : Cu, Fe, Zn → ÷ 1000
# mg/100mL : Vitamin C → × 10
# µg/100mL : Vitamin B군 → ÷ 100

# 존재하는 컬럼만 처리 (없는 경우 KeyError 방지)
for col in ["Cu_", "Fe_", "Zn_"]:
    if col in df.columns:
        df[col] = df[col] / 1000

if "Vitamin_C_" in df.columns:
    df["Vitamin_C_"] = df["Vitamin_C_"] * 10

for bvit in ["Vitamin_B1_", "Vitamin_B2_", "Vitamin_B3_", "Vitamin_B6_"]:
    if bvit in df.columns:
        df[bvit] = df[bvit] / 100

# === 6️⃣ 필요없는 컬럼 제거 및 순서 정리 ===
keep_cols = ["Water_type", "Tea_form", "Temperature", "Time_min"] + [
    col for col in df.columns if col not in ["TempTime", "Water_type", "Tea_form", "Temperature", "Time_min"]
]
df = df[keep_cols]

# === 7️⃣ 저장 ===
df.to_csv("green_tea_full_data_final.csv", index=False, encoding="utf-8-sig")
print("✅ 정제 완료: green_tea_full_data_final.csv 생성됨")
print("📊 최종 크기:", df.shape)
print(df.head())
