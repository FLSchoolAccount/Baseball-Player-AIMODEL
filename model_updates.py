"""Chronological rating models; legacy stats and unrelated tools remain available."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import data_logic as legacy

MODEL_VERSION = "reliability-v1"
HITTER_FEATURES = ["G", "AB", "H", "BB", "HR", "2B", "3B", "SO", "BA", "SLG", "OBP", "OPS"]
LEAGUE_FEATURES = ["League_BA", "League_OBP", "League_SLG", "League_OPS", "League_HR_per_AB"]
PITCHER_FEATURES = ["Age", "W", "L", "G", "GS", "CG", "SHO", "SV", "IPouts", "H", "ER", "BB", "SO", "GF", "Overall_ERA", "Overall_BAOpp"]


def __getattr__(name):
    return getattr(legacy, name)


def add_hitter_formula_ovr(df):
    df = df.copy()
    if df.empty:
        return df
    def num(col):
        return pd.to_numeric(df[col], errors="coerce").fillna(0)
    ab = num("AB").replace(0, np.nan)
    raw = 100 * (
        0.20 * num("OBP") + 0.09 * num("SLG")
        + 0.10 * (num("HR") / ab).fillna(0)
        + 0.06 * np.sqrt(num("HR").clip(lower=0))
        + 0.036 * np.sqrt(num("H").clip(lower=0))
        + 0.00015 * num("BB") - 0.0045 * num("SO") / 10
        + 0.08 * num("BA")
    )
    df["Formula_OVR"] = raw.clip(70, 99).round().astype(int)
    return df


def _league_context(year, batting, fielding):
    season = batting.loc[batting["yearID"] == year].copy()
    positions = fielding.loc[fielding["yearID"] == year].groupby(["playerID", "POS"], as_index=False)["G"].sum()
    if not positions.empty:
        primary = positions.sort_values(["playerID", "G", "POS"], ascending=[True, False, True]).drop_duplicates("playerID")
        pitchers = primary.loc[primary["POS"] == "P", "playerID"]
        season = season.loc[~season["playerID"].isin(pitchers)]
    cols = ["AB", "H", "BB", "HBP", "SF", "2B", "3B", "HR"]
    totals = season[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum()
    ab = totals["AB"]
    denom = ab + totals["BB"] + totals["HBP"] + totals["SF"]
    ba = totals["H"] / ab if ab else np.nan
    obp = (totals["H"] + totals["BB"] + totals["HBP"]) / denom if denom else np.nan
    tb = totals["H"] + totals["2B"] + 2 * totals["3B"] + 3 * totals["HR"]
    slg = tb / ab if ab else np.nan
    return dict(zip(LEAGUE_FEATURES, [ba, obp, slg, obp + slg, totals["HR"] / ab if ab else np.nan]))


def _hitter_seasons(start, end):
    batting = pd.read_csv(legacy.BATTING_CSV_PATH)
    fielding = pd.read_csv(legacy.FIELDING_CSV_PATH)
    people = pd.read_csv(legacy.PEOPLE_CSV_PATH)
    seasons = {}
    available = set(batting["yearID"].unique())
    for year in range(start, end + 1):
        if year in legacy.EXCLUDED_YEARS or year not in available:
            continue
        df = legacy.battingStats(year, batting_data=batting, fielding_data=fielding, people_data=people)
        if df.empty:
            continue
        df = add_hitter_formula_ovr(df)
        for col, value in _league_context(year, batting, fielding).items():
            df[col] = value
        seasons[year] = df
    return seasons


def _training_rows(seasons, features, start_year, target_year, window, hitter):
    if not isinstance(window, int) or window < 1:
        raise ValueError("potential_window must be a positive integer.")
    rows = []
    for year in range(start_year, target_year):
        if year not in seasons or year - 1 not in seasons:
            continue
        prev = seasons[year - 1][["playerID"] + features].rename(columns={c: "Prev_" + c for c in features})
        current_cols = ["playerID", "Age", "Formula_OVR"] if hitter else ["playerID", "Formula_OVR"]
        current = seasons[year][current_cols].rename(columns={"Formula_OVR": "OVR_Target"})
        frame = current.merge(prev, on="playerID", how="inner", validate="one_to_one")
        frame["Season"] = year
        frame["Projected_OVR_Target"] = np.nan
        frame["Potential_Target"] = np.nan
        if hitter and year + 1 < target_year and year + 1 in seasons:
            nxt = seasons[year + 1].set_index("playerID")["Formula_OVR"]
            frame["Projected_OVR_Target"] = frame["playerID"].map(nxt)
        future_years = list(range(year + 1, year + window + 1)) if hitter else list(range(year, year + window + 1))
        # The entire calendar window must have elapsed; individual missing seasons are not zero ratings.
        if max(future_years) < target_year and all(y in seasons for y in future_years):
            future = pd.concat([seasons[y][["playerID", "Formula_OVR"]] for y in future_years])
            peak = future.groupby("playerID")["Formula_OVR"].max()
            frame["Potential_Target"] = frame["playerID"].map(peak)
        rows.append(frame)
    if not rows:
        raise ValueError("No historical player-season pairs are available before this year.")
    result = pd.concat(rows, ignore_index=True)
    model_features = (["Age"] if hitter else []) + ["Prev_" + c for c in features]
    result = result.dropna(subset=model_features + ["OVR_Target"])
    if result.empty:
        raise ValueError("No complete historical feature rows are available.")
    return result, model_features


def build_hitter_training_data(target_year, start_year=1990, potential_window=3):
    seasons = _hitter_seasons(start_year - 1, target_year - 1)
    features = HITTER_FEATURES + LEAGUE_FEATURES
    training, model_features = _training_rows(seasons, features, start_year, target_year, potential_window, True)
    return training, HITTER_FEATURES.copy(), model_features


def _hitter_weights(frame):
    ab = pd.to_numeric(frame["Prev_AB"], errors="coerce").fillna(0)
    return 0.90 + 0.20 * (ab / 600).clip(0, 1)


def _fit(training, features, target, n_estimators, hitter):
    eligible = training.dropna(subset=features + [target])
    if eligible.empty:
        raise ValueError("Not enough pre-season history to train " + target + ". Try a later year.")
    settings = dict(n_estimators=n_estimators, random_state=legacy.RANDOM_STATE, min_samples_leaf=3 if hitter else 2, n_jobs=-1)
    if hitter:
        settings["max_features"] = 0.8
    model = RandomForestRegressor(**settings)
    model.fit(eligible[features], eligible[target], sample_weight=_hitter_weights(eligible) if hitter else None)
    return model, len(eligible)


def _predict(model, frame, features):
    return pd.Series(model.predict(frame[features]), index=frame.index).clip(70, 99).round().astype(int)


def build_hitter_ratings(target_year, selected_team=None, start_year=1990, potential_window=3, n_estimators=300):
    if target_year in legacy.EXCLUDED_YEARS:
        raise ValueError(str(target_year) + " is excluded because it was a shortened season.")
    seasons = _hitter_seasons(start_year - 1, target_year)
    features = HITTER_FEATURES + LEAGUE_FEATURES
    training, model_features = _training_rows(seasons, features, start_year, target_year, potential_window, True)
    current = seasons.get(target_year, pd.DataFrame())
    previous = seasons.get(target_year - 1, pd.DataFrame())
    if current.empty or previous.empty:
        raise ValueError("Eligible current-year and immediately preceding-year hitter data are required.")
    prev = previous[["playerID"] + features].rename(columns={c: "Prev_" + c for c in features})
    ratings = current.merge(prev, on="playerID", how="inner", validate="one_to_one").dropna(subset=model_features)
    if ratings.empty:
        raise ValueError("No hitters match to complete previous-season features.")
    counts = {}
    for output, target in [("OVR", "OVR_Target"), ("Projected_OVR", "Projected_OVR_Target"), ("Potential", "Potential_Target")]:
        model, counts[output] = _fit(training, model_features, target, n_estimators, True)
        ratings[output] = _predict(model, ratings, model_features)
    ratings["Potential"] = np.maximum(ratings["Potential"], ratings["OVR"])
    if selected_team:
        ratings = ratings.loc[ratings["Team"].fillna("").str.split("/").apply(lambda teams: selected_team in teams)]
    cols = ["playerID", "Name", "Team", "PrimaryPosition", "Age", "OVR", "Formula_OVR", "Projected_OVR", "Potential", "G", "AB", "H", "BB", "HR", "2B", "SO", "BA", "SLG", "OBP", "OPS"]
    result = ratings[cols].sort_values(["OVR", "Potential"], ascending=False).reset_index(drop=True)
    result.attrs.update(model_version=MODEL_VERSION, training_rows=counts, information_cutoff=target_year - 1)
    return result


def build_training_data(target_year, potential_window=3, pitching_csv_path=None, people_csv_path=None):
    pitching_csv_path = pitching_csv_path or legacy.PITCHING_CSV_PATH
    people_csv_path = people_csv_path or legacy.PEOPLE_CSV_PATH
    raw = pd.read_csv(pitching_csv_path)
    minimum = int(raw["yearID"].min())
    start = max(target_year - 10, minimum + 1)
    seasons = {}
    available = set(raw["yearID"].unique())
    for year in range(start - 1, target_year):
        if year not in available or year in legacy.EXCLUDED_YEARS:
            continue
        df = legacy.add_formula_ovr(legacy.pitchingStats(year, pitching_csv_path, people_csv_path))
        if not df.empty:
            seasons[year] = df
    training, _ = _training_rows(seasons, PITCHER_FEATURES, start, target_year, potential_window, False)
    training["Formula_OVR"] = training["OVR_Target"]
    return training, PITCHER_FEATURES.copy()


def build_pitcher_ratings(target_year, potential_window=3, n_estimators=300):
    if target_year in legacy.EXCLUDED_YEARS or target_year - 1 in legacy.EXCLUDED_YEARS:
        raise ValueError("This season requires an excluded shortened-season input or target.")
    training, features = build_training_data(target_year, potential_window)
    model_features = ["Prev_" + c for c in features]
    current = legacy.add_formula_ovr(legacy.pitchingStats(target_year))
    previous = legacy.add_formula_ovr(legacy.pitchingStats(target_year - 1))
    if current.empty or previous.empty:
        raise ValueError("Eligible current-year and immediately preceding-year pitcher data are required.")
    prev = previous[["playerID"] + features].rename(columns={c: "Prev_" + c for c in features})
    ratings = current.merge(prev, on="playerID", how="inner", validate="one_to_one").dropna(subset=model_features)
    if ratings.empty:
        raise ValueError("No pitchers match to previous-season features.")
    counts = {}
    for output, target in [("OVR", "OVR_Target"), ("Potential", "Potential_Target")]:
        model, counts[output] = _fit(training, model_features, target, n_estimators, False)
        ratings[output] = _predict(model, ratings, model_features)
    ratings["Potential"] = np.maximum(ratings["Potential"], ratings["OVR"])
    ratings["Potential_minus_OVR"] = ratings["Potential"] - ratings["OVR"]
    cols = ["playerID", "Name", "Team", "Age", "OVR", "Potential", "Potential_minus_OVR", "W", "L", "G", "GS", "CG", "SHO", "SV", "IPouts", "H", "ER", "BB", "SO", "BAOpp", "ERA", "GF"]
    result = ratings[cols].sort_values(["OVR", "Potential"], ascending=False).reset_index(drop=True)
    result.attrs.update(model_version=MODEL_VERSION, training_rows=counts, information_cutoff=target_year - 1)
    return result


def get_player_ovr_history(player_id, start_year, final_year):
    rows = []
    for year in range(start_year, final_year + 1):
        if year in legacy.EXCLUDED_YEARS:
            continue
        df = legacy.hittingStats(year)
        if df.empty:
            continue
        player = df.loc[df["playerID"] == player_id]
        if player.empty:
            continue
        player = add_hitter_formula_ovr(player)
        player["Season"] = year
        rows.append(player)
    if not rows:
        return pd.DataFrame()
    cols = ["Season", "Name", "Team", "Age", "G", "AB", "HR", "BA", "OBP", "SLG", "OPS", "Formula_OVR"]
    return pd.concat(rows, ignore_index=True)[cols].sort_values("Season").reset_index(drop=True)
