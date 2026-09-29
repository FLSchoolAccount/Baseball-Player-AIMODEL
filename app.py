"""
app.py

Streamlit front-end for the baseball OVR / Potential / Salary models,
plus a "Build Your Lineup" minigame.

All heavy lifting (pandas, sklearn) lives in data_logic.py -- this file
only handles page layout, navigation, user inputs, and displaying results.
"""

import random

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data_logic as dl

st.set_page_config(page_title="Baseball Player Ratings", layout="wide")

YEARS = list(range(1990, 2025))

if "page" not in st.session_state:
    st.session_state["page"] = "home"


def go_to(page_name):
    st.session_state["page"] = page_name


def back_home_button():
    st.button("Back to Home", on_click=go_to, args=("home",), key=f"back_home_{st.session_state['page']}")


# ============================================================
# HOME PAGE
# ============================================================

def render_home():
    st.title("Baseball Player Ratings & Projections")
    st.caption(
        "Random-forest models trained on Lahman batting/pitching/salary data. "
        "Pick a tool below to get started."
    )

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5, _ = st.columns(2)

    with col1:
        st.subheader("Pitcher Ratings")
        st.write("Predict OVR and Potential for pitchers in any season.")
        st.button("Open Pitcher Ratings", key="open_pitchers", on_click=go_to, args=("pitchers",))

    with col2:
        st.subheader("Hitter Ratings")
        st.write("Predict OVR, Projected OVR, and Potential for a team's hitters.")
        st.button("Open Hitter Ratings", key="open_hitters", on_click=go_to, args=("hitters",))

    with col3:
        st.subheader("Pitcher Salary & Contracts")
        st.write("Predict next-season salary and contract length for pitchers.")
        st.button("Open Salary & Contracts", key="open_salary", on_click=go_to, args=("salary",))

    with col4:
        st.subheader("Player OVR History")
        st.write("Look up a hitter's Formula OVR across seasons.")
        st.button("Open Player History", key="open_history", on_click=go_to, args=("history",))

    with col5:
        st.subheader("Build Your Lineup (Minigame)")
        st.write("Spin a random pool of hitters from a season and draft your own 9-man lineup.")
        st.button("Play Build Your Lineup", key="open_game", on_click=go_to, args=("game",), type="primary")


# ============================================================
# PITCHER RATINGS PAGE
# ============================================================

def render_pitchers():
    back_home_button()
    st.header("Pitcher OVR & Potential")
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
# HITTER RATINGS PAGE
# ============================================================

def render_hitters():
    back_home_button()
    st.header("Hitter OVR, Projected OVR & Potential")
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
# SALARY & CONTRACTS PAGE
# ============================================================

def render_salary():
    back_home_button()
    st.header("Pitcher Salary & Contract-Length Projections")
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
# PLAYER OVR HISTORY PAGE
# ============================================================

def render_history():
    back_home_button()
    st.header("Player OVR History")
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


# ============================================================
# MINIGAME: BUILD YOUR LINEUP
# ============================================================

LINEUP_SLOTS = ["1. Leadoff", "2. Contact", "3. Best Hitter", "4. Cleanup", "5. RBI Guy",
                 "6. Middle", "7. Middle", "8. Bottom", "9. Bottom"]

POOL_SIZE = 18


def _spin_new_pool(year):
    try:
        season_df = dl.hittingStats(year)
        if season_df.empty:
            return None, "No hitter data available for that year."

        season_df = dl.add_hitter_formula_ovr(season_df)
        pool_n = min(POOL_SIZE, len(season_df))
        pool = season_df.sample(n=pool_n, random_state=random.randint(0, 999999)).reset_index(drop=True)
        return pool, None
    except Exception as error:
        return None, str(error)


def _grade_for_score(avg_ovr):
    if avg_ovr >= 90:
        return "World Series Contender", "\U0001F3C6"
    elif avg_ovr >= 84:
        return "Playoff Team", "\u2B50"
    elif avg_ovr >= 78:
        return "Middle of the Pack", "\u26BE"
    elif avg_ovr >= 73:
        return "Rebuilding", "\U0001F527"
    else:
        return "Bottom Feeder", "\U0001FAAB"


def render_game():
    back_home_button()
    st.header("Build Your Lineup")
    st.write(
        "Spin a random pool of hitters from a season, then draft a 9-man lineup. "
        "Your team's grade is based on the average Formula OVR of your picks."
    )

    game_year = st.selectbox("Season to draft from", YEARS, index=len(YEARS) - 1, key="game_year")

    col_spin, col_reset = st.columns(2)
    with col_spin:
        spin_clicked = st.button("Spin New Player Pool", type="primary", key="spin_pool")
    with col_reset:
        reset_clicked = st.button("Reset Lineup", key="reset_lineup")

    if spin_clicked:
        with st.spinner(f"Spinning a pool of {POOL_SIZE} hitters from {game_year}..."):
            pool, error = _spin_new_pool(game_year)
            if error:
                st.error(f"Could not build a player pool: {error}")
            else:
                st.session_state["game_pool"] = pool
                st.session_state["game_pool_year"] = game_year
                st.session_state["lineup"] = {slot: None for slot in LINEUP_SLOTS}

    if reset_clicked and "lineup" in st.session_state:
        st.session_state["lineup"] = {slot: None for slot in LINEUP_SLOTS}

    if "game_pool" not in st.session_state:
        st.info("Click 'Spin New Player Pool' to deal your first set of hitters.")
        return

    pool = st.session_state["game_pool"]
    pool_year = st.session_state["game_pool_year"]

    if "lineup" not in st.session_state:
        st.session_state["lineup"] = {slot: None for slot in LINEUP_SLOTS}

    st.subheader(f"Available Hitters ({pool_year})")
    display_pool = pool[["Name", "Team", "Age", "G", "AB", "HR", "BA", "OBP", "SLG", "OPS", "Formula_OVR"]].copy()
    display_pool = display_pool.rename(columns={"Formula_OVR": "OVR"})
    st.dataframe(display_pool, use_container_width=True, height=300)

    name_to_row = {row["Name"]: row for _, row in pool.iterrows()}
    player_name_options = ["-- empty --"] + sorted(name_to_row.keys())

    st.subheader("Your Lineup")
    lineup = st.session_state["lineup"]

    used_names = {v for v in lineup.values() if v is not None}

    cols = st.columns(3)
    for i, slot in enumerate(LINEUP_SLOTS):
        with cols[i % 3]:
            current = lineup.get(slot)
            default_index = player_name_options.index(current) if current in player_name_options else 0
            chosen = st.selectbox(slot, player_name_options, index=default_index, key=f"slot_{slot}")
            lineup[slot] = None if chosen == "-- empty --" else chosen

    st.session_state["lineup"] = lineup

    filled = [v for v in lineup.values() if v is not None]
    duplicate_names = {name for name in filled if filled.count(name) > 1}

    if duplicate_names:
        st.warning(f"You picked the same player in more than one slot: {', '.join(duplicate_names)}. Pick different players for each spot.")
    elif len(filled) < 9:
        st.info(f"{len(filled)}/9 slots filled. Fill every slot to see your team grade.")
    else:
        picked_rows = [name_to_row[name] for name in filled]
        picked_df = pd.DataFrame(picked_rows)[["Name", "Team", "Age", "HR", "BA", "OBP", "SLG", "OPS", "Formula_OVR"]]
        picked_df = picked_df.rename(columns={"Formula_OVR": "OVR"})

        avg_ovr = picked_df["OVR"].mean()
        avg_ops = picked_df["OPS"].mean()
        total_hr = picked_df["HR"].sum()
        grade, emoji = _grade_for_score(avg_ovr)

        st.success("Lineup complete!")
        st.dataframe(picked_df, use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Average OVR", f"{avg_ovr:.1f}")
        m2.metric("Average OPS", f"{avg_ops:.3f}")
        m3.metric("Total HR", f"{int(total_hr)}")
        m4.metric("Team Grade", f"{emoji} {grade}")

        csv_bytes = picked_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download your lineup as CSV", csv_bytes, f"my_lineup_{pool_year}.csv", "text/csv")


# ============================================================
# ROUTER
# ============================================================

PAGES = {
    "home": render_home,
    "pitchers": render_pitchers,
    "hitters": render_hitters,
    "salary": render_salary,
    "history": render_history,
    "game": render_game,
}

PAGES.get(st.session_state["page"], render_home)()
