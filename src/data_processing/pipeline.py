import pandas as pd
import numpy as np
from typing import Tuple

# Usamos la clase EloSystem definida previamente, importándola localmente si es necesario, 
# pero la pasamos como parámetro para mantener el pipeline puro
def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Carga el CSV, limpia valores nulos y ordena por fecha."""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=["date"], ascending=True, inplace=True)
    df.dropna(subset=["home_score", "away_score"], inplace=True)
    return df

def apply_elo_system(df: pd.DataFrame, elo_system) -> pd.DataFrame:
    """Itera sobre los partidos para calcular y registrar el ELO de cada equipo."""
    elo_home_list = []
    elo_away_list = []
    
    for row in df.itertuples():
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        tournament = str(row.tournament)
        
        old_local_elo = elo_system.get_elo(home_team)
        old_away_elo = elo_system.get_elo(away_team)
        
        elo_home_list.append(old_local_elo)
        elo_away_list.append(old_away_elo)
        
        is_neutral = getattr(row, 'neutral', False)
        
        local_advantage_points = 0.0
        away_advantage_points = 0.0
        
        if not is_neutral:
            if row.country == home_team:
                local_advantage_points = 100.0  
                away_advantage_points = -100.0  
            elif row.country == away_team:
                local_advantage_points = -100.0  
                away_advantage_points = 100.0

        expected_home = elo_system.get_expected_result(old_local_elo, old_away_elo, local_advantage_points)
        expected_away = elo_system.get_expected_result(old_away_elo, old_local_elo, away_advantage_points)
        
        goal_multiplier = elo_system.get_goal_multiplier(row.home_score, row.away_score)
        k_weight = elo_system.get_weight(tournament)
        
        result_home = elo_system.get_winner_result(row.home_score, row.away_score)
        result_away = elo_system.get_winner_result(row.away_score, row.home_score)

        elo_system.update_elo(home_team, goal_multiplier, k_weight, result_home, expected_home)
        elo_system.update_elo(away_team, goal_multiplier, k_weight, result_away, expected_away)

    df = df.copy()
    df['home_elo'] = elo_home_list
    df['away_elo'] = elo_away_list
    return df

def filter_relevant_games(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra los juegos relevantes (fechas y torneos)."""
    df = df[(df["tournament"] != "Friendly") | (df["date"] >= '2026-04-01')].copy()
    df = df[df["date"] >= '2000-01-01'].copy()
    
    relevant_cols = ["date", "home_team", "away_team", "home_score", "away_score", "home_elo", "away_elo", "country", "neutral"]
    relevant_games_set = df[relevant_cols].copy()
    relevant_games_set.dropna(inplace=True)
    
    relevant_games_set["winner"] = np.where(
        relevant_games_set["home_score"] > relevant_games_set["away_score"], 1,
        np.where(relevant_games_set["home_score"] == relevant_games_set["away_score"], 0, -1)
    )
    return relevant_games_set

def build_team_history(relevant_games_set: pd.DataFrame) -> pd.DataFrame:
    """Construye el historial por equipo calculando medias móviles y rachas."""
    renamed = relevant_games_set.rename(columns={
        "home_team": "team", 
        "away_team": "opponent", 
        "home_score": "goals",
        "away_score": "opponent_goals", 
        "home_elo": "elo", 
        "away_elo": "opponent_elo"
    }).copy()

    df_local = renamed[["date", "team", "opponent", "goals", "opponent_goals", "elo", "opponent_elo", "winner", "country"]].copy()
    df_local["is_home"] = np.where(df_local["team"] == df_local["country"], 1, 0)

    df_away = renamed[["date", "team", "opponent", "goals", "opponent_goals", "elo", "opponent_elo", "winner", "country"]].copy()
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

    team_history = pd.concat([df_local, df_away])
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
    """Combina el dataset de juegos con las estadísticas calculadas."""
    df_stats = team_history[["team", "date", "last_5_goals_average", "last_5_vsgoals_average", "5_streak", "elo_diff"]].copy()
    
    df_final = relevant_games_set[["date", "home_team", "away_team", "winner", "neutral", "country", "home_score", "away_score"]].copy()

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

def split_data(df_final: pd.DataFrame, training_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Separa en train, val, test y devuelve todos los objetos necesarios:
    x_train, x_test, y_reg_home_train, y_reg_away_train,
    y_reg_test_home, y_reg_test_away
    """
    df_train = df_final[df_final['date'] < '2025-01-01'].copy()
    df_test = df_final[df_final['date'] >= '2025-01-01'].copy()

    x_train = df_train[training_cols]

    y_regression_home_train = df_train['home_score']
    y_regression_away_train = df_train['away_score']



    x_test = df_test[training_cols]
    y_reg_test_home = df_test['home_score'].copy()
    y_reg_test_away = df_test['away_score'].copy()
    
    return (x_train, x_test, 
            y_regression_home_train, y_regression_away_train, 
            y_reg_test_home, y_reg_test_away)

def full_pipeline(file_path: str, elo_system) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta todo el pipeline y devuelve df_final (para training) y team_history (útil para predicciones).
    """
    df_raw = load_and_clean_data(file_path)
    df_elo = apply_elo_system(df_raw, elo_system)
    df_relevant = filter_relevant_games(df_elo)
    team_history = build_team_history(df_relevant)
    df_final = build_final_features(df_relevant, team_history)
    return df_final, team_history
