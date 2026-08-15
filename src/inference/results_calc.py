import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import StackingRegressor
from math import nan
from src.math_utils.EloSystem import EloSystem
from src.math_utils.stats_utils import process_match, get_chances, get_top_x_probabilities_array
from typing import Optional

def get_predictions(
    home_team: str, 
    away_team: str, 
    home_stack: StackingRegressor, 
    away_stack: StackingRegressor, 
    neutral: bool, 
    elo_sys: EloSystem, 
    team_history: pd.DataFrame, 
    config: Dict[str, Any], 
    iterations: int = 3,
    date:Optional[str] = None
) -> Dict[str, Any]:
    """Calculate match probabilities and most likely scorelines from preprocessed team state."""
    raw_rho: Optional[float] = config.get("constants", {}).get("RHO") or config.get("RHO")
 
    if raw_rho is None or np.isnan(raw_rho):
        raise KeyError("RHO value not found in configuration.")

    rho : float = float(raw_rho)

    home_stats = get_average_stats(home_team, date, team_history)
    away_stats = get_average_stats(away_team, date, team_history)

    if not home_stats or not away_stats:
        raise ValueError("One or more teams has not enough data in the specified date.") 

    training_data: Dict[str, float] = {
        "diff_goals_5_matches": home_stats["team_goal_avg"] - away_stats["team_goal_avg"],
        "diff_vsgoals_5_matches": home_stats["team_vsgoal_avg"] - away_stats["team_vsgoal_avg"],
        "diff_streak_5_matches": home_stats["team_streak_avg"] - away_stats["team_streak_avg"],
        "elo_diff": home_stats["elo"] - away_stats["elo"],
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


def get_average_stats(team: str, date: Optional[str], team_history: pd.DataFrame) -> Dict[str, float]:

    """Returns the elo and the average stats on the specified date, of the specified team. 
    If there's no date, returns the latest stats.
    If the date is further than the dataframe's date, latest stats will be retrieved.
    If there's no enough data from the specified date, the dictionary will be empty.
    """

    recent_history : pd.DataFrame = team_history[team_history["team"] == team]

    if date is not None:
        recent_history = recent_history[recent_history["date"] <= date].copy()

    if recent_history.empty:
        return {}
    
    elo : float = recent_history["elo"].iloc[-1]
    team_goal_avg: float = recent_history["last_5_goals_average"].iloc[-1]
    team_vsgoal_avg: float = recent_history["last_5_vsgoals_average"].iloc[-1]
    team_streak_avg: float = recent_history["5_streak"].iloc[-1]

    return {
        "elo": elo,
        "team_goal_avg": team_goal_avg,
        "team_vsgoal_avg": team_vsgoal_avg,
        "team_streak_avg": team_streak_avg,
    }