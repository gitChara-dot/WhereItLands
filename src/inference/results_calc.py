import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import StackingRegressor
from src.math_utils.EloSystem import EloSystem
from src.math_utils.stats_utils import process_match, get_chances, get_top_x_probabilities_array
from math import nan
def get_predictions(
    home_team: str, 
    away_team: str, 
    home_stack: StackingRegressor, 
    away_stack: StackingRegressor, 
    neutral: bool, 
    elo_sys: EloSystem, 
    team_history: pd.DataFrame, 
    config: Dict[str, Any], 
    iterations: int = 3
) -> Dict[str, Any]:
    """Calcula y retorna la prediccion de probabilidades y resultados mas probables de un partido."""
    rho: float = config.get("constants", {}).get("RHO") or config.get("RHO", nan)
    
    if rho is None or rho is nan:
        raise KeyError("El valor RHO no existe en la configuracion.")

    home_recent: pd.DataFrame = team_history[team_history["team"] == home_team]
    away_recent: pd.DataFrame = team_history[team_history["team"] == away_team]

    home_goal_avg: float = home_recent["last_5_goals_average"].iloc[-1]
    away_goal_avg: float = away_recent["last_5_goals_average"].iloc[-1]

    home_vsgoal_avg: float = home_recent["last_5_vsgoals_average"].iloc[-1]
    away_vsgoal_avg: float = away_recent["last_5_vsgoals_average"].iloc[-1]

    home_streak_avg: float = home_recent["5_streak"].iloc[-1]
    away_streak_avg: float = away_recent["5_streak"].iloc[-1]

    elo_diff: float = elo_sys.get_elo(home_team) - elo_sys.get_elo(away_team)

    training_data: Dict[str, float] = {
        "diff_goals_5_matches": home_goal_avg - away_goal_avg,
        "diff_vsgoals_5_matches": home_vsgoal_avg - away_vsgoal_avg,
        "diff_streak_5_matches": home_streak_avg - away_streak_avg,
        "elo_diff": elo_diff,
        "home_advantage": 0.0 if neutral else 1.0
    }

    training_df: pd.DataFrame = pd.DataFrame([training_data])

    results_home: np.ndarray = home_stack.predict(training_df)
    results_away: np.ndarray = away_stack.predict(training_df)

    lambda_home: float = float(results_home[0])
    lambda_away: float = float(results_away[0])

    match_matrix: np.ndarray = process_match(lambda_home, lambda_away, rho)
    chances: List[float] = get_chances(match_matrix)
    top_results: List[Tuple[int, int, float]] = get_top_x_probabilities_array(match_matrix, iterations)

    final_prediction: Dict[str, Any] = {
        "home_win_chance": chances[0],
        "draw_chance": chances[1],
        "away_win_chance": chances[2],
        "top_results": top_results
    }

    return final_prediction
