
import pandas as pd
import numpy as np
from sklearn.ensemble import StackingRegressor
from src.math_utils import EloSystem
from src.math_utils.stats_utils import process_match, get_chances, get_top_x_probabilities_array
from typing import Any
# Calculates the final match prediction. Asumes that the home team gets the advantage if neutral = true.
def get_predictions(
        home_team : str, away_team : str, home_stack : StackingRegressor, away_stack : StackingRegressor, 
        neutral : bool, elo_sys : EloSystem, team_history : pd.DataFrame, 
        config : dict, iterations : int = 3
    ) -> dict[str, Any]:

    RHO = config.get("constants",{}).get("RHO") or config.get("RHO")

    if RHO is None:
        raise KeyError("RHO value doesn't exist on config, or config is misconfigured.")
    
    home_recent = team_history[team_history["team"] == home_team]
    away_recent = team_history[team_history["team"] == away_team]

    home_goal_avg = home_recent["last_5_goals_average"].iloc[-1]
    away_goal_avg = away_recent["last_5_goals_average"].iloc[-1]

    home_vsgoal_avg = home_recent["last_5_vsgoals_average"].iloc[-1]
    away_vsgoal_avg = away_recent["last_5_vsgoals_average"].iloc[-1]

    home_streak_avg = home_recent["5_streak"].iloc[-1]
    away_streak_avg = away_recent["5_streak"].iloc[-1]

    elo_diff = elo_sys.get_elo(home_team) - elo_sys.get_elo(away_team)

    training_data = {
        "diff_goals_5_matches" : home_goal_avg - away_goal_avg,
        "diff_vsgoals_5_matches" : home_vsgoal_avg - away_vsgoal_avg,
        "diff_streak_5_matches" : home_streak_avg - away_streak_avg, 
        "elo_diff": elo_diff, 
        "home_advantage" : 0 if neutral else 1
    }

    training_df = pd.DataFrame([training_data])

    results_home_dict : np.ndarray = home_stack.predict(training_df)
    results_away_dict : np.ndarray = away_stack.predict(training_df)

    lambda_home = results_home_dict[0]
    lambda_away = results_away_dict[0]

    match_matrix = process_match(lambda_home, lambda_away, RHO)
    chances = get_chances(match_matrix)

    top_results = get_top_x_probabilities_array(match_matrix, iterations)
    
    final_prediction : dict = {
        "home_win_chance" : chances[0],
        "draw_chance" : chances[1],
        "away_win_chance" : chances[2],
        "top_results" : top_results
    }

    return final_prediction






