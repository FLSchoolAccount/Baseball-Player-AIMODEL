"""
app.py

Streamlit front-end for the baseball OVR / Potential / Salary models.
All heavy lifting (pandas, sklearn) lives in data_logic.py -- this file
only handles page layout, user inputs, and displaying results.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data_logic as dl

st.set_page_config(page_title="Baseball Player Ratings", layout="wide")

st.title("Baseball Player Ratings & Projections")
st.caption(
    "Random-forest models trained on Lahman batting/pitching/salary data. "
    "Pick a tab, choose a year, and build the table."
)

YEARS = list(range(1990, 2025))

tab_pitchers, tab_hitters, tab_salary, tab_history = st.tabs(
    ["Pitcher Ratings", "Hitter Ratings", "Pitcher Salary & Contracts", "Player OVR History"]
)


# ============================================================
# TAB 1: PITCHER RATINGS
# ============================================================

with tab_pitchers:
    st.subheader("Pitcher OVR & Potential")
    st.write(
        "Trains a random forest on the surrounding 20 seasons of pitcher stats, "
        "then predicts this year's OVR and each pitcher's Potential (max OVR over the next 3 years)."
    )

    pitcher_year = st.selectbox("Year", YEARS, index=len(YEARS) - 1, key="pitcher_year")

    if st.button("Build Pitcher Ratings", type="primary", key="build_pitchers"):
        with st.spinner(f"Training models and building {pitcher_year} pitcher ratings..."):
            try:
                pitcher_df = dl.build_pitcher_ratings(pitcher_year)
                st.session_state["pitcher_df"] = pitcher_df
                st.session_state["pitcher_df_year"] = pitcher_year
            except Exception as error:
                st.error(f"Could not build pitcher ratings for {pitcher_year}: {error}")
                st.session_state.pop("pitcher_df", None)

    if "pitcher_df" in st.session_state:
        pdf = st.session_state["pitcher_df"]
        py = st.session_state["pitcher_df_year"]

        st.success(f"Built {len(pdf):,} pitcher ratings for {py}.")
        st.dataframe(pdf, use_container_width=True, height=420)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(pdf, x="OVR", nbins=30, title=f"{py} Pitcher OVR Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(pdf, x="Potential", nbins=30, title=f"{py} Pitcher Potential Distribution")
            st.plotly_chart(fig, use_container_width=True)

        csv_bytes = pdf.to_csv(index=False).encode("utf-8")
        st.download_button("Download pitcher ratings as CSV", csv_bytes, f"pitcher_ratings_{py}.csv", "text/csv")


# ============================================================
# TAB 2: HITTER RATINGS
# ============================================================

with tab_hitters:
    st.subheader("Hitter OVR, Projected OVR & Potential")
    st.write(
        "Pick a year, then a team, to build ratings for qualified hitters "
        "(non-pitchers with more than 70 games played)."
    )

    hitter_year = st.selectbox("Year", YEARS, index=len(YEARS) - 1, key="hitter_year")

    team_options = ["All teams"]
    try:
        year_df = dl.hittingStats(hitter_year)
        if not year_df.empty and "Team" in year_df.columns:
            teams = (
                year_df["Team"].dropna().astype(str).str.strip()
                .str.split("/").explode().str.strip()
            )
            teams = sorted(teams[teams != ""].unique().tolist())
            team_options = ["All teams"] + teams
    except Exception:
        pass

    hitter_team = st.selectbox("Team", team_options, key="hitter_team")

    if st.button("Build Hitter Ratings", type="primary", key="build_hitters"):
        selected_team = None if hitter_team == "All teams" else hitter_team
        with st.spinner(f"Training models and building {hitter_year} hitter ratings..."):
            try:
                hitter_df = dl.build_hitter_ratings(hitter_year, selected_team=selected_team)
                st.session_state["hitter_df"] = hitter_df
                st.session_state["hitter_df_year"] = hitter_year
                st.session_state["hitter_df_team"] = hitter_team
            except Exception as error:
                st.error(f"Could not build hitter ratings: {error}")
                st.session_state.pop("hitter_df", None)

    if "hitter_df" in st.session_state:
        hdf = st.session_state["hitter_df"]
        hy = st.session_state["hitter_df_year"]
        ht = st.session_state["hitter_df_team"]

        st.success(f"Built {len(hdf):,} hitter ratings for {ht} in {hy}.")
        st.dataframe(hdf, use_container_width=True, height=420)

        if "OPS" in hdf.columns and "OVR" in hdf.columns:
            fig = px.scatter(
                hdf, x="OPS", y="OVR", hover_data=["Name", "Team"],
                title=f"{hy} {ht} Hitters: OVR vs OPS",
            )
            st.plotly_chart(fig, use_container_width=True)

        csv_bytes = hdf.to_csv(index=False).encode("utf-8")
        st.download_button("Download hitter ratings as CSV", csv_bytes, f"hitter_ratings_{hy}_{ht}.csv", "text/csv")


# ============================================================
# TAB 3: PITCHER SALARY & CONTRACT LENGTH
# ============================================================

with tab_salary:
    st.subheader("Pitcher Salary & Contract-Length Projections")
    st.write(
        "First builds pitcher OVR/Potential ratings for the selected year, "
        "then trains a separate random forest on historical salary data to "
        "predict next-season salary and projected contract length."
    )

    salary_year = st.selectbox("Year", YEARS, index=len(YEARS) - 1, key="salary_year")

    if st.button("Build Salary Table", type="primary", key="build_salary"):
        with st.spinner(f"Building {salary_year} pitcher ratings and salary projections..."):
            try:
                pitcher_ratings = dl.build_pitcher_ratings(salary_year)
                pitcher_ratings = pitcher_ratings.rename(columns={"OVR": "Predicted_OVR"})

                current_stats = dl.add_formula_ovr(dl.pitchingStats(salary_year))
                merge_cols = [c for c in current_stats.columns if c not in pitcher_ratings.columns] + ["playerID"]
                pitcher_ratings_full = pitcher_ratings.merge(
                    current_stats[merge_cols], left_on="Name", right_on="Name", how="left"
                ) if "playerID" not in pitcher_ratings.columns else pitcher_ratings

                salary_df, _ = dl.build_salary_table(pitcher_ratings_full, salary_year)
                st.session_state["salary_df"] = salary_df
                st.session_state["salary_df_year"] = salary_year
            except Exception as error:
                st.error(f"Could not build salary table for {salary_year}: {error}")
                st.session_state.pop("salary_df", None)

    if "salary_df" in st.session_state:
        sdf = st.session_state["salary_df"]
        sy = st.session_state["salary_df_year"]

        st.success(f"Built {len(sdf):,} salary and contract-length projections for {sy}.")
        st.dataframe(sdf, use_container_width=True, height=420)

        csv_bytes = sdf.to_csv(index=False).encode("utf-8")
        st.download_button("Download salary projections as CSV", csv_bytes, f"pitcher_salary_{sy}.csv", "text/csv")


# ============================================================
# TAB 4: PLAYER OVR HISTORY
# ============================================================

with tab_history:
    st.subheader("Player OVR History")
    st.write("Look up a hitter's Formula OVR across seasons.")

    history_year = st.selectbox("Season to search up to", YEARS, index=len(YEARS) - 1, key="history_year")
    player_search = st.text_input("Player name contains", key="player_search", placeholder="e.g. Judge")

    if player_search:
        try:
            season_df = dl.hittingStats(history_year)
            matches = season_df[season_df["Name"].str.contains(player_search, case=False, na=False)]
            if matches.empty:
                st.info("No players matched that name for this season.")
            else:
                player_options = dict(zip(matches["Name"], matches["playerID"]))
                chosen_name = st.selectbox("Select player", list(player_options.keys()))

                if st.button("Show OVR History", type="primary", key="show_history"):
                    player_id = player_options[chosen_name]
                    history_df = dl.get_player_ovr_history(player_id, start_year=1990, final_year=history_year)

                    if history_df.empty:
                        st.warning(f"No historical OVR data found for {chosen_name}.")
                    else:
                        st.dataframe(history_df, use_container_width=True)

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=history_df["Season"], y=history_df["Formula_OVR"],
                            mode="lines+markers", name="Formula OVR",
                        ))
                        fig.update_layout(
                            title=f"{chosen_name} OVR Progression",
                            xaxis_title="Season", yaxis_title="OVR", yaxis_range=[69, 100],
                        )
                        st.plotly_chart(fig, use_container_width=True)
        except Exception as error:
            st.error(f"Could not search players: {error}")
