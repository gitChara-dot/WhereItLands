import pandas as pd
import numpy as np
from typing import Tuple, List
from src.math_utils.EloSystem import EloSystem


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Load raw match dataset, parse dates, sort chronologically, and drop missing scores."""
    df: pd.DataFrame = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=["date"], ascending=True, inplace=True)
    df.dropna(subset=["home_score", "away_score"], inplace=True)
    return df


def apply_elo_system(df: pd.DataFrame, elo_system: EloSystem) -> pd.DataFrame:
    """Iterate through historical matches sequentially to update and record team Elo ratings."""
    elo_home_list: List[float] = []
    elo_away_list: List[float] = []
    
    for row in df.itertuples():
        home_team: str = str(row.home_team)
        away_team: str = str(row.away_team)
        tournament: str = str(row.tournament)
        
        old_local_elo: float = elo_system.get_elo(home_team)
        old_away_elo: float = elo_system.get_elo(away_team)
        
        elo_home_list.append(old_local_elo)
        elo_away_list.append(old_away_elo)
        
        is_neutral: bool = getattr(row, 'neutral', False)
        
        local_advantage_points: float = 0.0
        away_advantage_points: float = 0.0
        
        if not is_neutral:
            if row.country == home_team:
                local_advantage_points = 100.0  
                away_advantage_points = -100.0  
            elif row.country == away_team:
                local_advantage_points = -100.0  
                away_advantage_points = 100.0

        expected_home: float = elo_system.get_expected_result(old_local_elo, old_away_elo, local_advantage_points)
        expected_away: float = elo_system.get_expected_result(old_away_elo, old_local_elo, away_advantage_points)
        
        home_score: int = int(row.home_score)  # type: ignore[arg-type]
        away_score: int = int(row.away_score)  # type: ignore[arg-type]

        goal_multiplier: float = elo_system.get_goal_multiplier(home_score, away_score)
        k_weight: int = elo_system.get_weight(tournament)
        
        result_home: float = elo_system.get_winner_result(home_score, away_score)
        result_away: float = elo_system.get_winner_result(away_score, home_score)

        elo_system.update_elo(home_team, goal_multiplier, k_weight, result_home, expected_home)
        elo_system.update_elo(away_team, goal_multiplier, k_weight, result_away, expected_away)

    df = df.copy()
    df['home_elo'] = elo_home_list
    df['away_elo'] = elo_away_list
    return df


def filter_relevant_games(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataset by date boundaries and competitive tournaments, encoding match winner."""
    df = df[(df["tournament"] != "Friendly") | (df["date"] >= '2026-04-01')].copy()
    df = df[df["date"] >= '2000-01-01'].copy()
    
    relevant_cols: List[str] = [
        "date", "home_team", "away_team", "home_score", "away_score", 
        "home_elo", "away_elo", "country", "neutral"
    ]
    relevant_games: pd.DataFrame = df[relevant_cols].copy()
    relevant_games.dropna(inplace=True)
    
    relevant_games["winner"] = np.where(
        relevant_games["home_score"] > relevant_games["away_score"], 1,
        np.where(relevant_games["home_score"] == relevant_games["away_score"], 0, -1)
    )
    return relevant_games


def build_team_history(relevant_games_set: pd.DataFrame) -> pd.DataFrame:
    """Transform matches into team-centric rows and compute rolling weighted averages and streaks."""
    renamed: pd.DataFrame = relevant_games_set.rename(columns={
        "home_team": "team", 
        "away_team": "opponent", 
        "home_score": "goals",
        "away_score": "opponent_goals", 
        "home_elo": "elo", 
        "away_elo": "opponent_elo"
    }).copy()

    df_local: pd.DataFrame = renamed[["date", "team", "opponent", "goals", "opponent_goals", "elo", "opponent_elo", "winner", "country"]].copy()
    df_local["is_home"] = np.where(df_local["team"] == df_local["country"], 1, 0)

    df_away: pd.DataFrame = renamed[["date", "team", "opponent", "goals", "opponent_goals", "elo", "opponent_elo", "winner", "country"]].copy()
    df_away.rename(inplace=True, columns={
        "team": "opponent", "opponent": "team", 
        "goals": "opponent_goals", "opponent_goals": "goals", 
        "elo": "opponent_elo", "opponent_elo": "elo"
    })
    df_away["winner"] = df_away["winner"] * -1
    df_away["is_home"] = np.where(df_away["team"] == df_away["country"], 1, 0)

    df_local.drop(columns=["country"], inplace=True)
    df_away.drop(columns=["country"], inplace=True)

    df_local.reset_index(inplace=True, drop=True)
    df_away.reset_index(inplace=True, drop=True)

    team_history: pd.DataFrame = pd.concat([df_local, df_away])
    team_history.sort_values(inplace=True, by=["team", "date"], ascending=[True, True])
    team_history.reset_index(inplace=True, drop=True)

    team_history['goal_multiplier'] = team_history['opponent_elo'] / 1500.0 
    team_history['goal_multiplier_defensive'] = 1500.0 / team_history['opponent_elo']
    team_history['ponderated_goal'] = team_history['goals'] * team_history['goal_multiplier']
    team_history['ponderated_received_goals'] = team_history['opponent_goals'] * team_history['goal_multiplier_defensive']

    team_history["last_5_goals_average"] = team_history.groupby('team')['ponderated_goal'].transform(
        lambda x: x.shift(1).rolling(5).mean()
    )
    team_history["last_5_vsgoals_average"] = team_history.groupby('team')['ponderated_received_goals'].transform(
        lambda x: x.shift(1).rolling(5).mean()
    )
    team_history["5_streak"] = team_history.groupby('team')['winner'].transform(
        lambda x: x.shift(1).rolling(5).sum()
    )

    team_history["elo_diff"] = team_history["elo"] - team_history["opponent_elo"]
    team_history.dropna(inplace=True) 
    
    return team_history


def build_final_features(relevant_games_set: pd.DataFrame, team_history: pd.DataFrame) -> pd.DataFrame:
    """Merge historical rolling statistics back into match records to produce final feature matrix."""
    df_stats: pd.DataFrame = team_history[["team", "date", "last_5_goals_average", "last_5_vsgoals_average", "5_streak", "elo_diff"]].copy()
    
    df_final: pd.DataFrame = relevant_games_set[["date", "home_team", "away_team", "winner", "neutral", "country", "home_score", "away_score"]].copy()

    df_final = pd.merge(left=df_final, right=df_stats, how='left', left_on=["date", "home_team"], right_on=["date", "team"])
    df_final = pd.merge(left=df_final, right=df_stats, how='left', left_on=["date", "away_team"], right_on=["date", "team"], suffixes=("_home", "_away"))

    df_final["diff_goals_5_matches"] = df_final["last_5_goals_average_home"] - df_final["last_5_goals_average_away"]
    df_final["diff_vsgoals_5_matches"] = df_final["last_5_vsgoals_average_home"] - df_final["last_5_vsgoals_average_away"]
    df_final["diff_streak_5_matches"] = df_final["5_streak_home"] - df_final["5_streak_away"]

    df_final.dropna(inplace=True)
    df_final.drop(columns=["team_home", "team_away", "elo_diff_away"], inplace=True)
    df_final.rename(inplace=True, columns={"elo_diff_home": "elo_diff"})
    
    df_final["home_advantage"] = np.where(df_final["neutral"] == True, 0, 1)
    df_final['winner'] = df_final['winner'].map({-1: 0, 0: 1, 1: 2})
    
    return df_final


def split_data(
    df_final: pd.DataFrame, 
    training_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split processed dataset into training and test partitions based on temporal cutoffs."""
    df_train: pd.DataFrame = df_final[df_final['date'] < '2025-01-01'].copy()
    df_test: pd.DataFrame = df_final[df_final['date'] >= '2025-01-01'].copy()

    x_train: pd.DataFrame = df_train[training_cols]
    y_regression_home_train: pd.Series = df_train['home_score']
    y_regression_away_train: pd.Series = df_train['away_score']

    x_test: pd.DataFrame = df_test[training_cols]
    y_reg_test_home: pd.Series = df_test['home_score'].copy()
    y_reg_test_away: pd.Series = df_test['away_score'].copy()
    
    return (
        x_train, x_test, 
        y_regression_home_train, y_regression_away_train, 
        y_reg_test_home, y_reg_test_away
    )


def full_pipeline(file_path: str, elo_system: EloSystem) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute end-to-end data preparation returning feature matrix and team history dataframe."""
    df_raw: pd.DataFrame = load_and_clean_data(file_path)
    df_elo: pd.DataFrame = apply_elo_system(df_raw, elo_system)
    df_relevant: pd.DataFrame = filter_relevant_games(df_elo)
    team_history: pd.DataFrame = build_team_history(df_relevant)
    df_final: pd.DataFrame = build_final_features(df_relevant, team_history)
    return df_final, team_history


def get_available_teams(team_history: pd.DataFrame) -> List[str]:
    """Return an alphabetically sorted list of unique team names from history records."""
    teams: List[str] = sorted(team_history["team"].unique().tolist())
    return teams