# Green Tea Extraction Optimizer / 녹차 추출 최적화 도구

## English
### Overview
- A Streamlit app that recommends brewing conditions (water type, tea form, temperature, time) to increase or decrease specific compounds (e.g., Na, Ca, Vitamin C). Taste is **not** modeled; only compound yields are considered.
- Single-page flow: constraints → compound objectives → top-3 recipes → explorer (2D trade-off, 3D view).

### Data & Columns
- Data file: `green_tea_full_data_final.csv` (expected in repo root).
- Condition columns (used as constraints): `Water_type`, `Tea_form`, `Temperature`, `Time_min`.
- Compound columns (targets): all other numeric columns.

### How it works
1) Load and clean data (column normalization, numeric casting).
2) Apply user constraints:
   - Allowed water types, tea forms (multi-select)
   - Max temperature (slider)
   - Max extraction time (slider)
3) User picks up to 3 compounds and direction (higher/lower).
4) Scoring (simple multi-objective heuristic):
   - Min-max normalize each selected compound.
   - Higher → normalized value; Lower → 1 - normalized value.
   - Average the utilities → sort → top-3 recommendations.
5) Explorer:
   - 2D trade-off scatter for the first two chosen compounds.
   - 3D scatter: Temperature × Time_min × chosen compound.

### Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Streamlit Cloud
- Repo: https://github.com/SeungDaniel/AdvancedProgramming
- App file: `streamlit_app.py`
- Branch: `main`
- After pushing changes: in Streamlit Cloud click “Clear cache and reboot” to reinstall deps and reload.
- https://brewingtea.streamlit.app/

### Files of interest
- `streamlit_app.py` — main app
- `requirements.txt` — deps (`streamlit`, `pandas`, `scikit-learn`, `plotly`)
- `green_tea_full_data_final.csv` — dataset

### Roadmap ideas
- Pareto-front display for more than two compounds
- Sensitivity analysis per compound vs. temperature/time
- Export recommended recipes as CSV

---

## 한국어
### 개요
- 특정 성분(예: Na, Ca, Vitamin C)을 늘리거나 줄이기 위해 추출 조건(물 종류, 티 형태, 온도, 시간)을 추천하는 Streamlit 앱입니다. **맛은 고려하지 않으며**, 성분 함량만을 기반으로 최적화합니다.
- 단일 페이지 흐름: 제약 입력 → 성분 목표 설정 → Top-3 레시피 → 고급 탐색(2D trade-off, 3D).

### 데이터와 컬럼
- 데이터 파일: `green_tea_full_data_final.csv` (리포지토리 루트에 위치)
- 조건 컬럼(제약에 사용): `Water_type`, `Tea_form`, `Temperature`, `Time_min`
- 성분 컬럼(목표): 나머지 모든 수치형 컬럼

### 동작 방식
1) 데이터 로드 및 컬럼 정리 (이름 정규화, 수치 변환 시도)
2) 제약 적용:
   - 사용 가능한 물/티 형태 선택
   - 최대 온도, 최대 추출 시간 슬라이더
3) 성분 목표 최대 3개 선택 (늘리기/줄이기)
4) 스코어링(단순 다중목표 휴리스틱):
   - 선택 성분별 min-max 정규화
   - 늘리기: 정규화값, 줄이기: 1 - 정규화값
   - 평균 유틸리티로 정렬 → Top-3 추천
5) Explorer:
   - 선택 성분 2개 기준 2D trade-off 산점도
   - 온도 × 시간 × 성분 3D 산점도

### 로컬 실행
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Streamlit Cloud
- 리포: https://github.com/SeungDaniel/AdvancedProgramming
- 앱 파일: `streamlit_app.py`
- 브랜치: `main`
- 변경 후 배포 갱신: Streamlit Cloud에서 “Clear cache and reboot” 클릭
- https://brewingtea.streamlit.app/

### 주요 파일
- `streamlit_app.py` — 메인 앱
- `requirements.txt` — 의존성 (`streamlit`, `pandas`, `scikit-learn`, `plotly`)
- `green_tea_full_data_final.csv` — 데이터셋

### 후속 개선 아이디어
- 두 개 이상 성분에 대한 파레토 프론트 표현
- 온도/시간 대비 성분 민감도 분석
- 추천 레시피 CSV 내보내기
