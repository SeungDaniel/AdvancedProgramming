import pandas as pd
'''
def clean_table(file_path):
    """각 Table 엑셀 파일을 자동 정리해주는 함수"""
    # 파일 불러오기 (3번째 행이 열 이름)
    df = pd.read_excel(file_path, header=2)
    
    # 열 이름 정리
    df.columns = (
        df.columns
        .str.replace(r"\(.*\)", "", regex=True)   # 괄호 속 단위 제거
        .str.strip()                              # 양쪽 공백 제거
        .str.replace(" ", "_")                    # 공백 → 언더바
        .str.replace(r"[^A-Za-z0-9_]+", "", regex=True)  # 특수문자 제거
    )

    # 숫자형 변환 (NaN 처리 포함)
    for col in df.columns[3:]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)   # ±, g, µ, c 등 비숫자 제거
            .replace("", None)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# 파일 경로 (이름은 예시)
minerals = clean_table("Minerals.xlsx")
polyphenols = clean_table("Polyphenols.xlsx")
vitamins = clean_table("Vitamins.xlsx")

#common columns
common_cols = ["Water_type", "Tea_form", "TempTime"]  # 표마다 다르면 여기 수정

# 병합 (공통 컬럼 기준, outer join으로 누락 방지)
merged = minerals.merge(polyphenols, on=common_cols, how="outer")
merged = merged.merge(vitamins, on=common_cols, how="outer")
merged.columns = merged.columns.str.replace(" ", "_")


merged.to_csv("green_tea_full_data.csv", index=False)
print("✅ green_tea_full_data.csv 저장 완료:", merged.shape)
'''

# === 파일 불러오기 ===
minerals = pd.read_excel("Minerals.xlsx", header=2)
polyphenols = pd.read_excel("Polyphenols.xlsx", header=2)
vitamins = pd.read_excel("Vitamins.xlsx", header=2)

# === 컬럼명 통일 ===
# 세 시트 모두 공통 기준 컬럼이 존재해야 함
minerals.rename(columns={"Water type": "Water_type", "Tea form": "Tea_form", "Temp./Time": "TempTime"}, inplace=True)
polyphenols.rename(columns={"Water type": "Water_type", "Tea form": "Tea_form", "Temp./Time": "TempTime"}, inplace=True)
vitamins.rename(columns={"Water type": "Water_type", "Tea form": "Tea_form", "Temp./Time": "TempTime"}, inplace=True)

# === 병합 (공통 조건 기준: Water_type, Tea_form, TempTime) ===
merged = minerals.merge(polyphenols, on=["Water_type", "Tea_form", "TempTime"], how="outer")
merged = merged.merge(vitamins, on=["Water_type", "Tea_form", "TempTime"], how="outer")

# === 결과 확인 ===
print("병합된 데이터 크기:", merged.shape)
print("열 이름 예시:", merged.columns.tolist()[:10])
print("상위 5행 미리보기:")
print(merged.head())

# === 저장 ===
merged.to_csv("green_tea_full_data.csv", index=False, encoding="utf-8-sig")
print("✅ green_tea_full_data.csv 파일이 생성되었습니다.")
