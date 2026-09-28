from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42
EXCLUDED_YEARS = {2020}

BATTING_CSV_PATH = "Batting.csv"
FIELDING_CSV_PATH = "Fielding.csv"
PEOPLE_CSV_PATH = "People.csv"
PITCHING_CSV_PATH = "Pitching.csv"
SALARIES_CSV_PATH = "Salaries.csv"

HITTER_STATS_CACHE = {}


# ============================================================
# SHARED HELPERS
# ============================================================

def _safe_int(value):
    try:
        return int(value)
    except Exception:
        return None


def _calculate_age(row, year):
    by = _safe_int(row.get("birthYear"))
    bm = _safe_int(row.get("birthMonth"))
    bd = _safe_int(row.get("birthDay"))

    if by is None or bm is None or bd is None:
        return pd.NA

    try:
        birth_date = datetime(by, bm, bd)
        ref_date = datetime(year, 7, 1)
        return ref_date.year - birth_date.year - (
            (ref_date.month, ref_date.day) < (birth_date.month, birth_date.day)
        )
    except Exception:
        return pd.NA


def _join_values(series):
    values = series.dropna().astype(str).str.strip()
    values = values[(values != "") & (values.str.lower() != "nan")]
    return "/".join(pd.unique(values))


# ============================================================
# BATTING STATS
# ============================================================

def battingStats(
    year,
    batting_csv_path=BATTING_CSV_PATH,
    fielding_csv_path=FIELDING_CSV_PATH,
    people_csv_path=PEOPLE_CSV_PATH,
    verbose=False,
    batting_data=None,
    fielding_data=None,
    people_data=None,
):
    batting = batting_data.copy() if batting_data is not None else pd.read_csv(batting_csv_path)
    fielding = fielding_data.copy() if fielding_data is not None else pd.read_csv(fielding_csv_path)
    people = people_data.copy() if people_data is not None else pd.read_csv(people_csv_path)

    batting = batting[batting["yearID"] == year].copy()
    fielding = fielding[fielding["yearID"] == year].copy()
    people = people.drop_duplicates(subset="playerID")

    if verbose:
        print(f"Number of batting records for year {year}: {len(batting)}")

    merged = batting.merge(people, on="playerID", how="left")
    merged["Name"] = (merged["nameFirst"].fillna("") + " " + merged["nameLast"].fillna("")).str.strip()
    merged["Age"] = merged.apply(lambda row: _calculate_age(row, year), axis=1)

    pos_counts = fielding.groupby(["playerID", "POS"], as_index=False)["G"].sum()

    if pos_counts.empty:
        primary_pos = pd.DataFrame(columns=["playerID", "PrimaryPosition"])
    else:
        primary_pos = (
            pos_counts.sort_values(["playerID", "G", "POS"], ascending=[True, False, True])
            .drop_duplicates(subset="playerID")
            .rename(columns={"POS": "PrimaryPosition"})[["playerID", "PrimaryPosition"]]
        )

    fielding_stats = fielding.groupby("playerID", as_index=False)[["PO", "A", "E", "DP"]].sum()

    batting_cols = ["playerID", "Name", "teamID", "Age", "G", "AB", "H", "BB", "HR", "2B", "3B", "SO", "HBP", "SF"]
    batting_df = merged[batting_cols].copy().rename(columns={"teamID": "Team"})

    sum_cols = ["G", "AB", "H", "BB", "HR", "2B", "3B", "SO", "HBP", "SF"]
    aggregations = {"Name": "first", "Team": _join_values, "Age": "first"}
    for col in sum_cols:
        aggregations[col] = "sum"

    batting_df = batting_df.merge(primary_pos, on="playerID", how="left")
    batting_df = batting_df.merge(fielding_stats, on="playerID", how="left")
    batting_df["PrimaryPosition"] = batting_df["PrimaryPosition"].fillna("DH")

    for col in ["PO", "A", "E", "DP"]:
        batting_df[col] = pd.to_numeric(batting_df[col], errors="coerce").fillna(0)

    aggregations = {
        "Team": _join_values,
        "PrimaryPosition": _join_values,
        "Age": "first",
        "G": "sum", "AB": "sum", "H": "sum", "BB": "sum", "HR": "sum",
        "2B": "sum", "3B": "sum", "SO": "sum", "HBP": "sum", "SF": "sum",
        "PO": "first", "A": "first", "E": "first", "DP": "first",
    }

    df = batting_df.groupby(["playerID", "Name"], as_index=False, dropna=False).agg(aggregations)
    df["PrimaryPosition"] = df["PrimaryPosition"].fillna("DH")

    for col in ["PO", "A", "E", "DP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["G"] = pd.to_numeric(df["G"], errors="coerce").fillna(0)
    df = df[df["PrimaryPosition"] != "P"].copy()
    df = df[df["G"] > 70].reset_index(drop=True)

    for col in ["AB", "BB", "HBP", "SF", "H", "2B", "3B", "HR", "SO"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["1B"] = df["H"] - df["2B"] - df["3B"] - df["HR"]
    df["TB"] = df["1B"] + 2 * df["2B"] + 3 * df["3B"] + 4 * df["HR"]

    ba_denom = df["AB"].replace(0, np.nan)
    obp_denom = df["AB"] + df["BB"] + df["HBP"] + df["SF"]

    df["BA"] = np.where(ba_denom.isna(), np.nan, df["H"] / ba_denom)
    df["OBP"] = np.where(obp_denom == 0, np.nan, (df["H"] + df["BB"] + df["HBP"]) / obp_denom)
    df["SLG"] = np.where(ba_denom.isna(), np.nan, df["TB"] / ba_denom)
    df["OPS"] = df["OBP"] + df["SLG"]

    fielding_denom = df["PO"] + df["A"] + df["E"]
    df["FldPct"] = np.where(fielding_denom == 0, 0, (df["PO"] + df["A"]) / fielding_denom)

    for col in ["BA", "OBP", "SLG", "OPS", "FldPct"]:
        df[col] = df[col].round(3)

    return df[[
        "playerID", "Name", "Team", "PrimaryPosition", "Age", "G", "AB", "H", "BB", "HR",
        "2B", "3B", "SO", "HBP", "SF", "PO", "A", "E", "DP", "FldPct", "1B", "TB", "BA", "OBP", "SLG", "OPS",
    ]]


def hittingStats(year, batting_csv_path=BATTING_CSV_PATH, fielding_csv_path=FIELDING_CSV_PATH,
                  people_csv_path=PEOPLE_CSV_PATH):
    if year in EXCLUDED_YEARS:
        return pd.DataFrame()

    if year not in HITTER_STATS_CACHE:
        HITTER_STATS_CACHE[year] = battingStats(
            year=year,
            batting_csv_path=batting_csv_path,
            fielding_csv_path=fielding_csv_path,
            people_csv_path=people_csv_path,
            verbose=False,
        )

    return HITTER_STATS_CACHE[year].copy()


# ============================================================
# PITCHING STATS
# ============================================================

def pitchingStats(year, pitching_csv_path=PITCHING_CSV_PATH, people_csv_path=PEOPLE_CSV_PATH,
                   min_games=15, min_starts=10):
    pitching = pd.read_csv(pitching_csv_path)
    people = pd.read_csv(people_csv_path)

    pitch_year = pitching.loc[pitching["yearID"] == year].copy()
    if pitch_year.empty:
        return pd.DataFrame()

    people_clean = people.drop_duplicates(subset="playerID").copy()
    merged = pitch_year.merge(people_clean, on="playerID", how="left", validate="many_to_one")
    merged["Name"] = (merged["nameFirst"].fillna("") + " " + merged["nameLast"].fillna("")).str.strip()
    merged["Age"] = merged.apply(lambda row: _calculate_age(row, year), axis=1)

    numeric_cols = ["W", "L", "G", "GS", "CG", "SHO", "SV", "IPouts", "H", "ER", "BB", "SO", "GF"]
    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    output_cols = ["playerID", "Name", "teamID", "Age"] + numeric_cols + ["BAOpp", "ERA"]
    pitching_df = merged[output_cols].copy().rename(columns={"teamID": "Team"})

    aggregations = {"Name": "first", "Team": _join_values, "Age": "first", "BAOpp": _join_values, "ERA": _join_values}
    for col in numeric_cols:
        aggregations[col] = "sum"

    pitching_df = pitching_df.groupby("playerID", as_index=False, dropna=False).agg(aggregations)
    pitching_df = pitching_df.loc[(pitching_df["G"] >= min_games) | (pitching_df["GS"] >= min_starts)].copy()

    pitching_df["Overall_ERA"] = np.where(
        pitching_df["IPouts"] == 0, np.nan, (pitching_df["ER"] * 27) / pitching_df["IPouts"]
    )
    pitching_df["Overall_BAOpp"] = np.where(
        pitching_df["IPouts"] == 0, np.nan,
        pitching_df["H"] / (pitching_df["IPouts"] / 3 + pitching_df["H"] + pitching_df["BB"]),
    )
    pitching_df["Overall_ERA"] = pitching_df["Overall_ERA"].round(3)
    pitching_df["Overall_BAOpp"] = pitching_df["Overall_BAOpp"].round(3)

    return pitching_df.reset_index(drop=True)


# ============================================================
# FORMULA OVR (HITTERS)
# ============================================================

def add_hitter_formula_ovr(df):
    df = df.copy()
    if df.empty:
        return df

    required_cols = ["G", "AB", "H", "BB", "HR", "2B", "3B", "SO", "BA", "SLG", "OBP", "OPS"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing hitter columns needed for Formula_OVR: {missing}")

    ab = pd.to_numeric(df["AB"], errors="coerce").replace(0, np.nan)
    hr = pd.to_numeric(df["HR"], errors="coerce").fillna(0)
    hits = pd.to_numeric(df["H"], errors="coerce").fillna(0)
    walks = pd.to_numeric(df["BB"], errors="coerce").fillna(0)
    strikeouts = pd.to_numeric(df["SO"], errors="coerce").fillna(0)
    ba = pd.to_numeric(df["BA"], errors="coerce").fillna(0)
    slg = pd.to_numeric(df["SLG"], errors="coerce").fillna(0)
    obp = pd.to_numeric(df["OBP"], errors="coerce").fillna(0)

    hr_per_ab = (hr / ab).fillna(0)
    sqrt_hr = np.sqrt(hr.clip(lower=0))
    sqrt_h = np.sqrt(hits.clip(lower=0))
    so_factor = strikeouts / 10

    if "E" in df.columns:
        errors = pd.to_numeric(df["E"], errors="coerce").fillna(0)
        e_component = errors - 0.95
    else:
        e_component = 0

    ovr_raw = (
        0.20 * obp + 0.09 * slg + 0.10 * hr_per_ab + 0.06 * sqrt_hr + 0.036 * sqrt_h
        + 0.00015 * walks - 0.0045 * so_factor + 0.08 * ba + 0.009 * e_component
    ) * 100

    df["Formula_OVR"] = pd.Series(ovr_raw, index=df.index).clip(lower=70, upper=99).round().astype(int)
    return df


# ============================================================
# FORMULA OVR (PITCHERS)
# ============================================================

def calculate_formula_ovr(row):
    era = row["Overall_ERA"]
    ba_opp = row["Overall_BAOpp"]

    if pd.isna(era) or pd.isna(ba_opp):
        return np.nan

    if row["GS"] >= row["GF"] * 5:
        rating = (
            0.14 * (4 - era) + 0.0002 * row["IPouts"] + 0.005 * row["SO"]
            + 0.6 * (row["SO"] / row["IPouts"] if row["IPouts"] != 0 else 0)
            + 0.01 * row["W"] + 0.1 * row["SHO"] - 0.00001 * row["BB"] - 0.1 * ba_opp
        ) * 100
    else:
        rating = (
            0.12 * (4.5 - era) + 0.006 * row["SO"]
            + 0.5 * (row["SO"] / row["IPouts"] if row["IPouts"] != 0 else 0)
            + 0.01 * row["SV"] + 0.005 * row["W"] + 0.01 * row["GF"]
            - 0.002 * row["BB"] - 0.1 * ba_opp
        ) * 100

    return np.clip(round(rating), 70, 99)


def add_formula_ovr(df):
    df = df.copy()
    if df.empty:
        return df

    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df.drop(columns=["OVR", "Formula_OVR", "Predicted_OVR", "Potential", "Potential_minus_OVR"], errors="ignore")

    required_cols = ["Age", "W", "L", "G", "GS", "CG", "SHO", "SV", "IPouts", "H", "ER", "BB", "SO", "GF",
                      "Overall_ERA", "Overall_BAOpp"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing pitcher columns: {missing}")

    df = df.dropna(subset=required_cols).copy()
    if df.empty:
        return df

    formula_ovr_values = df.apply(calculate_formula_ovr, axis=1)
    df["Formula_OVR"] = (
        pd.Series(formula_ovr_values.to_numpy(), index=df.index).clip(lower=70, upper=99).round().astype(int)
    )
    return df


# ============================================================
# PITCHER TRAINING DATA + RATINGS
# ============================================================

def build_training_data(target_year, potential_window=3, pitching_csv_path=PITCHING_CSV_PATH,
                         people_csv_path=PEOPLE_CSV_PATH):
    pitching_raw = pd.read_csv(pitching_csv_path)
    min_year = int(pitching_raw["yearID"].min())
    max_year = int(pitching_raw["yearID"].max())

    start_year = max(target_year - 10, min_year + 1)
    end_year = min(target_year + 10, max_year - potential_window)
    required_start_year = max(start_year - 1, min_year)
    required_end_year = min(end_year + potential_window, max_year)

    season_data = []
    for year in range(required_start_year, required_end_year + 1):
        season_df = pitchingStats(year, pitching_csv_path, people_csv_path)
        if season_df.empty:
            continue
        season_df = add_formula_ovr(season_df)
        if season_df.empty:
            continue
        season_df["Season"] = year
        season_data.append(season_df)

    if not season_data:
        raise ValueError("No usable pitching data exists in the required year range.")

    all_seasons = pd.concat(season_data, ignore_index=True)
    all_seasons = all_seasons.loc[:, ~all_seasons.columns.duplicated()].copy()

    feature_cols = ["Age", "W", "L", "G", "GS", "CG", "SHO", "SV", "IPouts", "H", "ER", "BB", "SO", "GF",
                     "Overall_ERA", "Overall_BAOpp"]

    previous_season = all_seasons[["playerID", "Season"] + feature_cols].copy()
    previous_season["Season"] = previous_season["Season"] + 1
    previous_season = previous_season.rename(columns={col: f"Prev_{col}" for col in feature_cols})

    current_season = all_seasons[["playerID", "Season", "Formula_OVR"]].copy()

    future_ovr_parts = []
    for years_ahead in range(potential_window + 1):
        future_part = all_seasons[["playerID", "Season", "Formula_OVR"]].copy()
        future_part["Season"] = future_part["Season"] - years_ahead
        future_part = future_part.rename(columns={"Formula_OVR": f"Future_OVR_{years_ahead}"})
        future_ovr_parts.append(future_part[["playerID", "Season", f"Future_OVR_{years_ahead}"]])

    future_ovr_df = future_ovr_parts[0]
    for part in future_ovr_parts[1:]:
        future_ovr_df = future_ovr_df.merge(part, on=["playerID", "Season"], how="outer", validate="one_to_one")

    future_ovr_cols = [f"Future_OVR_{i}" for i in range(potential_window + 1)]
    future_ovr_df["Potential_Target"] = future_ovr_df[future_ovr_cols].max(axis=1)

    training_df = current_season.merge(previous_season, on=["playerID", "Season"], how="inner", validate="one_to_one")
    training_df = training_df.merge(
        future_ovr_df[["playerID", "Season", "Potential_Target"]], on=["playerID", "Season"],
        how="inner", validate="one_to_one",
    )
    training_df = training_df.loc[(training_df["Season"] >= start_year) & (training_df["Season"] <= end_year)].copy()

    model_feature_cols = [f"Prev_{col}" for col in feature_cols]
    training_df = training_df.dropna(subset=model_feature_cols + ["Formula_OVR", "Potential_Target"]).copy()

    if training_df.empty:
        raise ValueError("No usable training rows were created for this year.")

    training_df["Formula_OVR"] = training_df["Formula_OVR"].clip(lower=70, upper=99).round().astype(int)
    training_df["Potential_Target"] = training_df["Potential_Target"].clip(lower=70, upper=99).round().astype(int)

    return training_df, feature_cols


def build_pitcher_ratings(target_year, potential_window=3, n_estimators=300):
    training_df, feature_cols = build_training_data(target_year, potential_window=potential_window)
    training_df = training_df.loc[:, ~training_df.columns.duplicated()].copy()

    model_features = [f"Prev_{col}" for col in feature_cols]
    X_train = training_df[model_features]

    ovr_model = RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, min_samples_leaf=2, n_jobs=-1)
    potential_model = RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, min_samples_leaf=2, n_jobs=-1)

    ovr_model.fit(X_train, training_df["Formula_OVR"])
    potential_model.fit(X_train, training_df["Potential_Target"])

    current_pitching = add_formula_ovr(pitchingStats(target_year))
    previous_pitching = add_formula_ovr(pitchingStats(target_year - 1))

    if current_pitching.empty:
        raise ValueError(f"No eligible pitchers were found for {target_year}.")
    if previous_pitching.empty:
        raise ValueError(f"No eligible pitchers were found for {target_year - 1}.")

    current_pitching = current_pitching.loc[:, ~current_pitching.columns.duplicated()].copy()
    previous_pitching = previous_pitching.loc[:, ~previous_pitching.columns.duplicated()].copy()

    previous_features = previous_pitching[["playerID"] + feature_cols].copy()
    previous_features = previous_features.rename(columns={col: f"Prev_{col}" for col in feature_cols})

    ratings_df = current_pitching.merge(previous_features, on="playerID", how="inner", validate="one_to_one")
    ratings_df = ratings_df.loc[:, ~ratings_df.columns.duplicated()].copy()
    ratings_df = ratings_df.dropna(subset=model_features).copy()

    if ratings_df.empty:
        raise ValueError("No pitchers matched to previous-season features.")

    ratings_df["Predicted_OVR"] = (
        pd.Series(ovr_model.predict(ratings_df[model_features]), index=ratings_df.index)
        .clip(lower=70, upper=99).round().astype(int)
    )
    ratings_df["Potential"] = (
        pd.Series(potential_model.predict(ratings_df[model_features]), index=ratings_df.index)
        .clip(lower=70, upper=99).round().astype(int)
    )
    ratings_df["Potential"] = np.maximum(ratings_df["Potential"], ratings_df["Predicted_OVR"])
    ratings_df["Potential_minus_OVR"] = ratings_df["Potential"] - ratings_df["Predicted_OVR"]

    final_df = ratings_df[[
        "Name", "Team", "Age", "Predicted_OVR", "Potential", "Potential_minus_OVR",
        "W", "L", "G", "GS", "CG", "SHO", "SV", "IPouts", "H", "ER", "BB", "SO", "BAOpp", "ERA", "GF",
    ]].copy().rename(columns={"Predicted_OVR": "OVR"})

    final_df = final_df.loc[:, ~final_df.columns.duplicated()].copy()
    return final_df.sort_values(["OVR", "Potential"], ascending=[False, False]).reset_index(drop=True)


# ============================================================
# HITTER TRAINING DATA + RATINGS
# ============================================================

def build_hitter_training_data(target_year, start_year=1990, potential_window=3):
    feature_cols = ["G", "AB", "H", "BB", "HR", "2B", "3B", "SO", "BA", "SLG", "OBP", "OPS"]
    required_cols = ["playerID", "Name", "Team", "Age"] + feature_cols

    seasons = {}
    for year in range(start_year - 1, target_year + potential_window + 1):
        if year in EXCLUDED_YEARS:
            continue
        try:
            season_df = hittingStats(year)
        except Exception:
            continue
        if season_df is None or season_df.empty:
            continue

        season_df = season_df.loc[:, ~season_df.columns.duplicated()].copy()
        missing = [c for c in required_cols if c not in season_df.columns]
        if missing:
            continue

        season_df = season_df.dropna(subset=["playerID", "Age"] + feature_cols).copy()
        if season_df.empty:
            continue

        season_df = add_hitter_formula_ovr(season_df)
        season_df["Season"] = year

        ab = pd.to_numeric(season_df["AB"], errors="coerce").replace(0, np.nan)
        season_df["League_BA"] = pd.to_numeric(season_df["BA"], errors="coerce").mean()
        season_df["League_OBP"] = pd.to_numeric(season_df["OBP"], errors="coerce").mean()
        season_df["League_SLG"] = pd.to_numeric(season_df["SLG"], errors="coerce").mean()
        season_df["League_OPS"] = pd.to_numeric(season_df["OPS"], errors="coerce").mean()
        season_df["League_HR_per_AB"] = (
            (pd.to_numeric(season_df["HR"], errors="coerce") / ab).replace([np.inf, -np.inf], np.nan).mean()
        )

        seasons[year] = season_df

    if len(seasons) < 3:
        raise ValueError("Not enough usable seasons were loaded. Check battingStats(), playerID, and your CSV paths.")

    training_rows = []
    for year in range(start_year, target_year):
        previous_year = year - 1
        if year in EXCLUDED_YEARS or previous_year in EXCLUDED_YEARS:
            continue
        if previous_year not in seasons or year not in seasons:
            continue
        if year + 1 in EXCLUDED_YEARS:
            continue

        previous_df = seasons[previous_year].copy()
        current_df = seasons[year].copy()

        previous_features = previous_df[
            ["playerID"] + feature_cols + ["League_BA", "League_OBP", "League_SLG", "League_OPS", "League_HR_per_AB"]
        ].copy()
        previous_features = previous_features.rename(
            columns={c: f"Prev_{c}" for c in previous_features.columns if c != "playerID"}
        )

        current_targets = current_df[["playerID", "Age", "Formula_OVR"]].copy().rename(
            columns={"Formula_OVR": "OVR_Target"}
        )
        current_targets["Season"] = year

        future_frames = []
        for future_year in range(year + 1, year + potential_window + 1):
            if future_year in EXCLUDED_YEARS or future_year not in seasons:
                continue
            future_part = seasons[future_year][["playerID", "Formula_OVR"]].copy()
            future_part["Future_Season"] = future_year
            future_frames.append(future_part)

        if not future_frames:
            continue

        future_df = pd.concat(future_frames, ignore_index=True)

        projected_ovr_target = future_df.loc[
            future_df["Future_Season"] == year + 1, ["playerID", "Formula_OVR"]
        ].copy()
        if projected_ovr_target.empty:
            continue
        projected_ovr_target = projected_ovr_target.rename(columns={"Formula_OVR": "Projected_OVR_Target"})

        potential_target = (
            future_df.groupby("playerID", as_index=False)["Formula_OVR"].max()
            .rename(columns={"Formula_OVR": "Potential_Target"})
        )

        year_training = current_targets.merge(previous_features, on="playerID", how="inner", validate="one_to_one")
        year_training = year_training.merge(projected_ovr_target, on="playerID", how="inner", validate="one_to_one")
        year_training = year_training.merge(potential_target, on="playerID", how="inner", validate="one_to_one")

        training_rows.append(year_training)

    if not training_rows:
        raise ValueError("No hitter training rows were created.")

    training_df = pd.concat(training_rows, ignore_index=True)

    model_features = (
        ["Age"] + [f"Prev_{c}" for c in feature_cols]
        + ["Prev_League_BA", "Prev_League_OBP", "Prev_League_SLG", "Prev_League_OPS", "Prev_League_HR_per_AB"]
    )

    training_df = training_df.dropna(
        subset=model_features + ["OVR_Target", "Projected_OVR_Target", "Potential_Target"]
    ).copy()

    return training_df, feature_cols, model_features


def get_player_ovr_history(player_id, start_year, final_year):
    history_rows = []
    for historical_year in range(start_year, final_year + 1):
        if historical_year in EXCLUDED_YEARS:
            continue
        try:
            season_df = hittingStats(historical_year)
        except Exception:
            continue
        if season_df is None or season_df.empty or "playerID" not in season_df.columns:
            continue

        player_season = season_df.loc[season_df["playerID"] == player_id].copy()
        if player_season.empty:
            continue

        player_season = add_hitter_formula_ovr(player_season)
        player_season["Season"] = historical_year

        history_rows.append(
            player_season[["Season", "Name", "Team", "Age", "G", "AB", "HR", "BA", "OBP", "SLG", "OPS", "Formula_OVR"]]
        )

    if not history_rows:
        return pd.DataFrame()

    return pd.concat(history_rows, ignore_index=True).sort_values("Season").reset_index(drop=True)


def build_hitter_ratings(target_year, selected_team=None, start_year=1990, potential_window=3, n_estimators=300):
    if target_year in EXCLUDED_YEARS:
        raise ValueError(f"{target_year} is excluded because it was a shortened season.")

    training_df, feature_cols, model_features = build_hitter_training_data(
        target_year=target_year, start_year=start_year, potential_window=potential_window
    )
    training_df = training_df.loc[training_df["Season"] < target_year].copy()

    if training_df.empty:
        raise ValueError(f"No training data exists before {target_year}.")

    X_train = training_df[model_features]
    model_settings = dict(n_estimators=n_estimators, random_state=RANDOM_STATE, min_samples_leaf=3,
                           max_features=0.8, n_jobs=-1)

    ovr_model = RandomForestRegressor(**model_settings)
    projected_ovr_model = RandomForestRegressor(**model_settings)
    potential_model = RandomForestRegressor(**model_settings)

    ovr_model.fit(X_train, training_df["OVR_Target"])
    projected_ovr_model.fit(X_train, training_df["Projected_OVR_Target"])
    potential_model.fit(X_train, training_df["Potential_Target"])

    current_hitting = hittingStats(target_year)
    previous_hitting = hittingStats(target_year - 1)

    if current_hitting.empty:
        raise ValueError(f"No current hitter data found for {target_year}.")
    if previous_hitting.empty:
        raise ValueError(f"No previous hitter data found for {target_year - 1}.")

    current_hitting = add_hitter_formula_ovr(current_hitting)
    previous_hitting = add_hitter_formula_ovr(previous_hitting)

    previous_ab = pd.to_numeric(previous_hitting["AB"], errors="coerce").replace(0, np.nan)
    previous_hitting["League_BA"] = pd.to_numeric(previous_hitting["BA"], errors="coerce").mean()
    previous_hitting["League_OBP"] = pd.to_numeric(previous_hitting["OBP"], errors="coerce").mean()
    previous_hitting["League_SLG"] = pd.to_numeric(previous_hitting["SLG"], errors="coerce").mean()
    previous_hitting["League_OPS"] = pd.to_numeric(previous_hitting["OPS"], errors="coerce").mean()
    previous_hitting["League_HR_per_AB"] = (
        (pd.to_numeric(previous_hitting["HR"], errors="coerce") / previous_ab)
        .replace([np.inf, -np.inf], np.nan).mean()
    )

    previous_features = previous_hitting[
        ["playerID"] + feature_cols + ["League_BA", "League_OBP", "League_SLG", "League_OPS", "League_HR_per_AB"]
    ].copy()
    previous_features = previous_features.rename(
        columns={c: f"Prev_{c}" for c in previous_features.columns if c != "playerID"}
    )

    ratings_df = current_hitting.merge(previous_features, on="playerID", how="inner", validate="one_to_one")
    ratings_df = ratings_df.dropna(subset=model_features).copy()

    if ratings_df.empty:
        raise ValueError("No players had matching previous-season features.")

    ratings_df["OVR"] = (
        pd.Series(ovr_model.predict(ratings_df[model_features]), index=ratings_df.index)
        .clip(lower=70, upper=99).round().astype(int)
    )
    ratings_df["Projected_OVR"] = (
        pd.Series(projected_ovr_model.predict(ratings_df[model_features]), index=ratings_df.index)
        .clip(lower=70, upper=99).round().astype(int)
    )
    ratings_df["Potential"] = (
        pd.Series(potential_model.predict(ratings_df[model_features]), index=ratings_df.index)
        .clip(lower=70, upper=99).round().astype(int)
    )
    ratings_df["Potential"] = np.maximum(ratings_df["Potential"], ratings_df["OVR"])

    final_columns = [
        "playerID", "Name", "Team", "PrimaryPosition", "Age", "OVR", "Formula_OVR", "Projected_OVR", "Potential",
        "G", "AB", "H", "BB", "HR", "2B", "SO", "BA", "SLG", "OBP", "OPS",
    ]
    final_columns = [c for c in final_columns if c in ratings_df.columns]
    final_df = ratings_df[final_columns].copy()

    if selected_team is not None:
        team_mask = (
            final_df["Team"].fillna("").astype(str).str.split("/")
            .apply(lambda team_list: selected_team in [t.strip() for t in team_list])
        )
        final_df = final_df.loc[team_mask].copy()

    if final_df.empty:
        raise ValueError(f"No qualified hitters were found for {selected_team} in {target_year}.")

    return final_df.sort_values(
        by=["OVR", "Potential", "Projected_OVR"], ascending=[False, False, False]
    ).reset_index(drop=True)


# ============================================================
# SALARY + CONTRACT-LENGTH MODEL (PITCHERS)
# ============================================================

def build_salary_table(pitching_df, target_year, salaries_csv_path=SALARIES_CSV_PATH,
                        salary_window=10, max_contract_years=8):
    if pitching_df.empty:
        raise ValueError("No pitcher rows were supplied for salary prediction.")

    required_pitching_cols = [
        "playerID", "Name", "Team", "Age", "Predicted_OVR", "Potential", "W", "L", "G", "GS", "CG", "SHO", "SV",
        "IPouts", "H", "ER", "BB", "SO", "GF", "Overall_ERA", "Overall_BAOpp",
    ]
    missing_pitching_cols = [c for c in required_pitching_cols if c not in pitching_df.columns]
    if missing_pitching_cols:
        raise ValueError(f"Pitching salary table is missing columns: {missing_pitching_cols}")

    salaries = pd.read_csv(salaries_csv_path)
    salaries_clean = salaries.copy()
    salaries_clean["yearID"] = pd.to_numeric(salaries_clean["yearID"], errors="coerce")
    salaries_clean["salary"] = pd.to_numeric(salaries_clean["salary"], errors="coerce")
    salaries_clean = salaries_clean.dropna(subset=["playerID", "yearID", "salary"]).copy()

    if salaries_clean.empty:
        raise ValueError("Salaries.csv contains no usable salary rows.")

    salaries_clean["yearID"] = salaries_clean["yearID"].astype(int)
    salaries_clean = salaries_clean.groupby(["playerID", "yearID"], as_index=False)["salary"].sum()

    salary_min_year = int(salaries_clean["yearID"].min())
    salary_max_year = int(salaries_clean["yearID"].max())
    salary_start_year = max(target_year - salary_window, salary_min_year + 1)
    salary_end_year = min(target_year + salary_window, salary_max_year - 1)

    if salary_start_year > salary_end_year:
        raise ValueError(
            f"Not enough salary history exists near {target_year}. "
            f"Salary data covers {salary_min_year}-{salary_max_year}."
        )

    salary_feature_cols = ["Age", "W", "L", "G", "GS", "CG", "SHO", "SV", "IPouts", "H", "ER", "BB", "SO", "GF",
                            "Overall_ERA", "Overall_BAOpp"]

    rating_seasons = []
    for year in range(salary_start_year - 1, salary_end_year + 1):
        season_stats = pitchingStats(year)
        if season_stats.empty:
            continue
        season_stats = add_formula_ovr(season_stats).copy()
        if season_stats.empty:
            continue
        season_stats["Season"] = year
        season_stats["Salary_Formula_OVR"] = season_stats["Formula_OVR"].clip(lower=70, upper=99).round().astype(int)
        rating_seasons.append(
            season_stats[["playerID", "Season", "Age", "Salary_Formula_OVR"] + salary_feature_cols[1:]].copy()
        )

    if not rating_seasons:
        raise ValueError("No historical pitcher-stat rows were available for salary-model training.")

    historical_stats = pd.concat(rating_seasons, ignore_index=True)

    current_salary = salaries_clean.rename(columns={"yearID": "Season", "salary": "Current_Salary"})
    next_salary = salaries_clean.copy()
    next_salary["yearID"] = next_salary["yearID"] - 1
    next_salary = next_salary.rename(columns={"yearID": "Season", "salary": "Next_Season_Salary"})

    league_salary = (
        salaries_clean.groupby("yearID", as_index=False)["salary"]
        .agg(League_Median_Salary="median", League_Mean_Salary="mean")
        .sort_values("yearID").reset_index(drop=True)
    )
    league_salary["Median_5Y_Ago"] = league_salary["League_Median_Salary"].shift(5)
    league_salary["Inflation_5Y"] = np.where(
        league_salary["Median_5Y_Ago"] > 0,
        league_salary["League_Median_Salary"] / league_salary["Median_5Y_Ago"], 1.0,
    )
    league_salary["Inflation_5Y"] = league_salary["Inflation_5Y"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    league_salary = league_salary.rename(columns={"yearID": "Season"})

    salary_years_by_player = salaries_clean.groupby("playerID")["yearID"].apply(lambda y: set(y.astype(int))).to_dict()

    def calculate_contract_duration(player_id, season):
        player_years = salary_years_by_player.get(player_id, set())
        duration = 0
        for years_ahead in range(1, max_contract_years + 1):
            if season + years_ahead in player_years:
                duration += 1
            else:
                break
        return duration

    training_df = historical_stats.merge(current_salary, on=["playerID", "Season"], how="inner", validate="one_to_one")
    training_df = training_df.merge(next_salary, on=["playerID", "Season"], how="inner", validate="one_to_one")
    training_df = training_df.merge(league_salary, on="Season", how="left", validate="many_to_one")

    training_df["Contract_Duration_Target"] = training_df.apply(
        lambda row: calculate_contract_duration(row["playerID"], int(row["Season"])), axis=1
    )

    training_features = salary_feature_cols + [
        "Salary_Formula_OVR", "Current_Salary", "League_Median_Salary", "League_Mean_Salary", "Inflation_5Y",
    ]

    training_df = training_df.dropna(
        subset=training_features + ["Next_Season_Salary", "Contract_Duration_Target"]
    ).copy()

    if training_df.empty:
        raise ValueError(
            "No historical salary-training rows remained after matching pitcher stats to consecutive salary seasons."
        )

    training_df["Log_Next_Season_Salary"] = np.log1p(training_df["Next_Season_Salary"])

    salary_model = RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, min_samples_leaf=3,
                                          max_features=0.8, n_jobs=-1)
    salary_model.fit(training_df[training_features], training_df["Log_Next_Season_Salary"])

    duration_model = RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, min_samples_leaf=3,
                                            max_features=0.8, n_jobs=-1)
    duration_model.fit(training_df[training_features], training_df["Contract_Duration_Target"])

    target_salaries = (
        salaries_clean.loc[salaries_clean["yearID"] == target_year]
        .groupby("playerID", as_index=False)["salary"].sum().rename(columns={"salary": "Salary"})
    )

    target_league_context = league_salary.loc[league_salary["Season"] == target_year].copy()
    if target_league_context.empty:
        target_league_context = league_salary.tail(1).copy()
    target_league_context = target_league_context.iloc[0]

    salary_output = pitching_df.copy()
    salary_output = salary_output.merge(target_salaries, on="playerID", how="left", validate="one_to_one")
    salary_output["Salary"] = pd.to_numeric(salary_output["Salary"], errors="coerce").fillna(0)
    salary_output["Salary_Formula_OVR"] = pd.to_numeric(salary_output["Predicted_OVR"], errors="coerce").fillna(70)
    salary_output["Current_Salary"] = salary_output["Salary"]
    salary_output["League_Median_Salary"] = target_league_context["League_Median_Salary"]
    salary_output["League_Mean_Salary"] = target_league_context["League_Mean_Salary"]
    salary_output["Inflation_5Y"] = target_league_context["Inflation_5Y"]

    for col in training_features:
        if col not in salary_output.columns:
            salary_output[col] = 0
        salary_output[col] = pd.to_numeric(salary_output[col], errors="coerce").fillna(0)

    salary_output["Predicted_Salary"] = np.expm1(salary_model.predict(salary_output[training_features]))
    salary_output["Predicted_Salary"] = salary_output["Predicted_Salary"].clip(lower=400000).round().astype("Int64")

    salary_output["Projected_Contract_Years"] = (
        pd.Series(duration_model.predict(salary_output[training_features]), index=salary_output.index)
        .clip(lower=1, upper=max_contract_years).round().astype(int)
    )

    salary_output["Salary"] = salary_output["Salary"].round().astype("Int64")
    salary_output["Salary_Difference"] = (salary_output["Predicted_Salary"] - salary_output["Salary"]).astype("Int64")

    salary_output = salary_output[[
        "Name", "Team", "Age", "Predicted_OVR", "Potential", "Salary", "Predicted_Salary",
        "Projected_Contract_Years", "Salary_Difference",
    ]].rename(columns={"Predicted_OVR": "OVR"})

    return salary_output.sort_values("Predicted_Salary", ascending=False).reset_index(drop=True), training_df


# ============================================================
# NEXT-SEASON HITTER STAT MODEL
# ============================================================

def build_hitter_next_stat_rfr(start_year=1990, end_year=2023, potential_window=3,
                                rating_n_estimators=300, stat_n_estimators=400):
    training_rows = []

    feature_columns = ["Age", "OVR", "Formula_OVR", "Projected_OVR", "Potential", "G", "AB", "H", "BB", "HR",
                        "2B", "SO", "BA", "OBP", "SLG", "OPS"]
    target_columns = ["Next_H", "Next_HR", "Next_BA", "Next_OBP", "Next_SLG", "Next_OPS"]

    for year in range(start_year, end_year):
        next_year = year + 1
        if year in EXCLUDED_YEARS or next_year in EXCLUDED_YEARS:
            continue

        try:
            current_ratings = build_hitter_ratings(
                target_year=year, selected_team=None, start_year=start_year,
                potential_window=potential_window, n_estimators=rating_n_estimators,
            ).copy()
            next_stats = hittingStats(next_year).copy()
        except Exception:
            continue

        if current_ratings.empty or next_stats.empty:
            continue

        required_current = ["playerID"] + feature_columns
        required_next = ["playerID", "H", "HR", "BA", "OBP", "SLG", "OPS"]

        if any(c not in current_ratings.columns for c in required_current):
            continue
        if any(c not in next_stats.columns for c in required_next):
            continue

        current_df = current_ratings[required_current].copy()
        next_df = next_stats[required_next].copy().rename(columns={
            "H": "Next_H", "HR": "Next_HR", "BA": "Next_BA", "OBP": "Next_OBP", "SLG": "Next_SLG", "OPS": "Next_OPS",
        })

        training_year = current_df.merge(next_df, on="playerID", how="inner", validate="one_to_one")
        if training_year.empty:
            continue

        training_year["Season"] = year
        training_rows.append(training_year)

    if not training_rows:
        raise ValueError("No valid hitter seasons were available for next-season stat-model training.")

    stat_training_df = pd.concat(training_rows, ignore_index=True)
    numeric_columns = feature_columns + target_columns

    for column in numeric_columns:
        stat_training_df[column] = pd.to_numeric(stat_training_df[column], errors="coerce")

    stat_training_df = stat_training_df.dropna(subset=numeric_columns).copy()
    if stat_training_df.empty:
        raise ValueError("No usable training rows remain after numeric cleaning.")

    stat_rfr = RandomForestRegressor(n_estimators=stat_n_estimators, random_state=RANDOM_STATE,
                                      min_samples_leaf=3, max_features=0.8, n_jobs=-1)
    stat_rfr.fit(stat_training_df[feature_columns], stat_training_df[target_columns])

    return stat_rfr, feature_columns, target_columns, stat_training_df
