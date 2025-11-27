# streamlit_app.py
"""
Green tea extraction optimizer.
Single page with: constraints → objectives → recommendations → explorer.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ===========================================
# ⚙️ Config
# ===========================================
st.set_page_config(
    page_title="Green Tea Extraction Optimizer",
    page_icon="🍵",
    layout="wide",
)


# ===========================================
# 🧰 Data helpers
# ===========================================
def load_data(csv_path: str = "green_tea_full_data_final.csv") -> pd.DataFrame:
    """
    Load CSV and apply basic cleanup.
    - Strips/normalizes column names
    - Attempts numeric casting where possible
    """
    df = pd.read_csv(csv_path)
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace(r"[()/%]", "", regex=True)
    )

    # Attempt numeric casting for non-object columns
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                # leave as-is for categorical columns
                pass
    return df


def get_condition_columns(df: pd.DataFrame) -> List[str]:
    """
    Return known condition columns (intersection with dataframe).
    Adjust the defaults if actual column names differ.
    """
    preferred = ["Water_type", "Tea_form", "Temperature", "Time_min"]
    return [c for c in preferred if c in df.columns]


def get_component_columns(df: pd.DataFrame, condition_cols: List[str]) -> List[str]:
    """
    Infer component (target) columns as numeric columns excluding condition columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in condition_cols]


# ===========================================
# 🧪 Filtering & objectives
# ===========================================
def filter_by_constraints(
    df: pd.DataFrame,
    allowed_water: List[str],
    allowed_form: List[str],
    max_temp: Optional[float],
    max_time: Optional[float],
) -> pd.DataFrame:
    """
    Apply user constraints to the dataset and return feasible rows.
    """
    filtered = df.copy()
    if allowed_water and "Water_type" in filtered.columns:
        filtered = filtered[filtered["Water_type"].isin(allowed_water)]
    if allowed_form and "Tea_form" in filtered.columns:
        filtered = filtered[filtered["Tea_form"].isin(allowed_form)]
    if max_temp is not None and "Temperature" in filtered.columns:
        filtered = filtered[filtered["Temperature"] <= max_temp]
    if max_time is not None and "Time_min" in filtered.columns:
        filtered = filtered[filtered["Time_min"] <= max_time]
    return filtered


def build_objectives_from_user_input(
    component_options: List[str],
) -> List[Tuple[str, str]]:
    """
    Collect up to 3 component objectives from user input.
    Returns list of (component, direction) where direction in {'high', 'low'}.
    """
    st.subheader("2️⃣ Which compounds do you want to increase/decrease?")
    st.markdown(
        "This tool recommends recipes that yield **more/less** of the selected compounds. "
        "Taste is not modeled."
    )

    objectives: List[Tuple[str, str]] = []
    cols = st.columns(3)
    for idx, col in enumerate(cols, start=1):
        with col:
            choice = st.selectbox(
                f"Compound {idx}",
                options=["Skip"] + component_options,
                index=0,
                key=f"component_{idx}",
            )
            if choice != "Skip":
                direction = st.radio(
                    "Direction",
                    options=["Higher", "Lower"],
                    horizontal=True,
                    key=f"direction_{idx}",
                )
                objectives.append((choice, "high" if direction == "Higher" else "low"))

    if not objectives:
        st.warning("Select at least one compound (e.g., lower Na, higher Ca).")
    return objectives


def compute_scores_for_objectives(
    df: pd.DataFrame, objectives: List[Tuple[str, str]]
) -> pd.Series:
    """
    Compute a simple average utility score per row based on objectives.
    - Min-max normalize each selected component
    - 'high': use normalized value
    - 'low': use 1 - normalized value
    """
    if not objectives or df.empty:
        return pd.Series([np.nan] * len(df), index=df.index)

    scores = []
    for comp, direction in objectives:
        if comp not in df.columns:
            continue
        col = df[comp].astype(float)
        comp_norm = (col - col.min()) / (col.max() - col.min() + 1e-9)
        if direction == "low":
            comp_norm = 1 - comp_norm
        scores.append(comp_norm)

    if not scores:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.concat(scores, axis=1).mean(axis=1)


def recommend_recipes(
    df: pd.DataFrame, objectives: List[Tuple[str, str]], top_n: int = 3
) -> pd.DataFrame:
    """
    Return top-N recipes sorted by computed score.
    """
    if df.empty or not objectives:
        return pd.DataFrame()
    scores = compute_scores_for_objectives(df, objectives)
    result = df.copy()
    result["__score"] = scores
    result = result.sort_values("__score", ascending=False)
    return result.head(top_n)


# ===========================================
# 📈 Visualization
# ===========================================
def plot_tradeoff_2d(
    df: pd.DataFrame,
    objectives: List[Tuple[str, str]],
    top_df: pd.DataFrame,
) -> Optional[go.Figure]:
    """2D scatter for two selected components (if available)."""
    if len(objectives) < 2:
        return None
    x_comp, y_comp = objectives[0][0], objectives[1][0]
    if x_comp not in df.columns or y_comp not in df.columns:
        return None
    fig = px.scatter(df, x=x_comp, y=y_comp, title="2D Trade-off")
    if not top_df.empty:
        fig.add_trace(
            go.Scatter(
                x=top_df[x_comp],
                y=top_df[y_comp],
                mode="markers",
                marker=dict(color="red", size=10, symbol="star"),
                name="Recommended",
            )
        )
    return fig


def plot_3d_surface(
    df: pd.DataFrame,
    z_comp: str,
    top_df: pd.DataFrame,
) -> Optional[go.Figure]:
    """3D scatter for Temperature-Time-Component."""
    required = {"Temperature", "Time_min", z_comp}
    if not required.issubset(df.columns):
        return None
    fig = px.scatter_3d(
        df,
        x="Temperature",
        y="Time_min",
        z=z_comp,
        color=z_comp,
        title=f"3D explorer: {z_comp}",
    )
    if not top_df.empty:
        fig.add_trace(
            go.Scatter3d(
                x=top_df["Temperature"],
                y=top_df["Time_min"],
                z=top_df[z_comp],
                mode="markers",
                marker=dict(color="red", size=6, symbol="diamond"),
                name="Recommended",
            )
        )
    return fig


# ===========================================
# 🚀 App Layout
# ===========================================
def main():
    st.title("🍵 Green Tea Extraction Optimizer")
    st.caption("Health-profile optimizer based on compound yields (taste not modeled)")

    df = load_data()
    condition_cols = get_condition_columns(df)
    component_cols = get_component_columns(df, condition_cols)

    # 1) Constraints
    st.header("1️⃣ Tell us your constraints")
    c1, c2 = st.columns(2)
    allowed_water = []
    allowed_form = []
    with c1:
        if "Water_type" in df.columns:
            water_options = sorted(df["Water_type"].dropna().unique().tolist())
            allowed_water = st.multiselect(
                "Select allowed water types",
                options=water_options,
                default=water_options,
            )
    with c2:
        if "Tea_form" in df.columns:
            form_options = sorted(df["Tea_form"].dropna().unique().tolist())
            allowed_form = st.multiselect(
                "Select allowed tea forms (Leaf, Powder, etc.)",
                options=form_options,
                default=form_options,
            )

    max_temp = None
    if "Temperature" in df.columns:
        max_temp = st.slider(
            "Maximum usable temperature (°C)",
            min_value=float(df["Temperature"].min()),
            max_value=float(df["Temperature"].max()),
            value=float(df["Temperature"].max()),
            step=1.0,
        )
    max_time = None
    if "Time_min" in df.columns:
        max_time = st.slider(
            "Maximum extraction time (min)",
            min_value=float(df["Time_min"].min()),
            max_value=float(df["Time_min"].max()),
            value=float(df["Time_min"].max()),
            step=1.0,
        )

    filtered_df = filter_by_constraints(df, allowed_water, allowed_form, max_temp, max_time)
    st.info(f"Feasible combinations: {len(filtered_df)} / {len(df)}")
    if filtered_df.empty:
        st.warning("No combinations satisfy your constraints. Please adjust selections.")
        return

    # 2) Compound objectives
    objectives = build_objectives_from_user_input(component_cols)
    if not objectives:
        return

    # 3) Recommendations
    scored = filtered_df.copy()
    scored["__score"] = compute_scores_for_objectives(filtered_df, objectives)
    top3 = recommend_recipes(filtered_df, objectives, top_n=3)

    st.header("3️⃣ Best recipes within your constraints")
    if top3.empty:
        st.warning("No recommendations. Adjust constraints or compound choices.")
    else:
        avg_series = filtered_df.mean(numeric_only=True)
        st.write("Top-3 recipes by score (based on selected compounds)")
        cards = st.columns(len(top3))
        for idx, (_, row) in enumerate(top3.iterrows()):
            with cards[idx]:
                st.markdown(f"#### 🏅 Recommendation #{idx + 1}")
                st.markdown(
                    f"- Water: **{row.get('Water_type', 'N/A')}**  \n"
                    f"- Tea form: **{row.get('Tea_form', 'N/A')}**  \n"
                    f"- Temperature: **{row.get('Temperature', 'N/A')} °C**  \n"
                    f"- Time: **{row.get('Time_min', 'N/A')} min**"
                )
                st.markdown("**Compound summary**")
                lines = []
                for comp, direction in objectives:
                    if comp not in row or comp not in avg_series:
                        continue
                    val = row[comp]
                    base = avg_series.get(comp, np.nan)
                    delta_pct = (val - base) / (base + 1e-9) * 100 if pd.notna(base) else np.nan
                    dir_text = "Increase" if direction == "high" else "Decrease"
                    lines.append(f"- {comp}: {val:.3f} ({delta_pct:+.1f}% vs. mean, goal: {dir_text})")
                if lines:
                    st.markdown("\n".join(lines))
                st.caption("Best fit for your constraints and compound goals.")

    # Explorer section
    with st.expander("Explorer (for research/deep dive)"):
        st.caption("Inspect trade-offs and condition–compound relationships.")
        fig2d = plot_tradeoff_2d(filtered_df, objectives, top3)
        if fig2d:
            st.plotly_chart(fig2d, use_container_width=True)
        else:
            st.info("Select at least two compounds to see a 2D trade-off.")

        if objectives:
            z_comp = objectives[0][0]
            fig3d = plot_3d_surface(filtered_df, z_comp, top3)
            if fig3d:
                st.plotly_chart(fig3d, use_container_width=True)
            else:
                st.info("Temperature, Time_min, and the chosen compound are needed for 3D.")


if __name__ == "__main__":
    main()
