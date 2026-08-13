import numpy as np
from scipy.stats import poisson
from typing import List, Tuple, Dict


def get_coles_dixon_correction(
    home_goals: int, 
    away_goals: int, 
    lambda_home: float, 
    lambda_away: float, 
    rho: float
) -> float:
    """Apply the Coles-Dixon bivariate adjustment factor for low-scoring match outcomes."""
    x = home_goals
    y = away_goals
    if x == 0 and y == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if x == 0 and y == 1:
        return 1.0 + lambda_home * rho
    if x == 1 and y == 0:
        return 1.0 + lambda_away * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def process_match(lambda_home: float, lambda_away: float, rho: float) -> np.ndarray:
    """Compute the 6x6 score probability matrix using Poisson PMFs and Coles-Dixon correction."""
    probability_matrix = np.zeros((6, 6))
    for score_home in range(6):
        for score_away in range(6):
            prob_home = poisson.pmf(score_home, lambda_home)
            prob_away = poisson.pmf(score_away, lambda_away)
            accumulated_prob = prob_home * prob_away
            if (score_home + score_away) <= 2:
                accumulated_prob *= get_coles_dixon_correction(score_home, score_away, lambda_home, lambda_away, rho)
            probability_matrix[score_home, score_away] = accumulated_prob
    
    return probability_matrix


def get_chances(probability_matrix: np.ndarray) -> List[float]:
    """Calculate aggregate win, draw, and loss probabilities from the probability matrix."""
    home_win_chance = 0.0
    away_win_chance = 0.0
    draw_chance = 0.0
    
    for i in range(probability_matrix.shape[0]):
        for j in range(probability_matrix.shape[1]):
            current_chance = probability_matrix[i][j]
            if i > j:
                home_win_chance += current_chance
            elif i == j:
                draw_chance += current_chance
            else:
                away_win_chance += current_chance
                
    return [home_win_chance, draw_chance, away_win_chance]


def get_likely_result_str(probability_matrix: np.ndarray) -> str:
    """Return the single most probable scoreline as a formatted string 'home-away'."""
    index = np.unravel_index(np.argmax(probability_matrix), probability_matrix.shape)
    return f"{index[0]}-{index[1]}"


def get_likely_result_tuple(probability_matrix: np.ndarray) -> List[int]:
    """Return the single most probable scoreline as a list of integers [home, away]."""
    index = np.unravel_index(np.argmax(probability_matrix), probability_matrix.shape)
    return [int(index[0]), int(index[1])]


def get_top_x_probabilities(probability_matrix: np.ndarray, iterations: int) -> Dict[str, float]:
    """Return the top N most probable scorelines as a dictionary mapping 'home-away' to percentage."""
    flattened_matrix = probability_matrix.flatten()
    if iterations <= 0 or iterations >= len(flattened_matrix):
        return {}
    
    top_x = np.argsort(flattened_matrix)[-iterations:][::-1]
    indexes = np.unravel_index(top_x, probability_matrix.shape)

    top_probabilities: Dict[str, float] = {}
    for i in range(iterations):
        row = indexes[0][i]
        col = indexes[1][i]
        prob = float(probability_matrix[row, col] * 100.0)
        coord_str = f"{row}-{col}"
        top_probabilities[coord_str] = prob
    
    return top_probabilities


def get_top_x_probabilities_array(probability_matrix: np.ndarray, iterations: int) -> List[Tuple[int, int, float]]:
    """Return the top N most probable scorelines as a list of tuples (home_goals, away_goals, percentage)."""
    flattened_matrix = probability_matrix.flatten()
    if iterations <= 0 or iterations >= len(flattened_matrix):
        return []
    
    top_x = np.argsort(flattened_matrix)[-iterations:][::-1]
    indexes = np.unravel_index(top_x, probability_matrix.shape)

    top_results_array: List[Tuple[int, int, float]] = []
    for i in range(iterations):
        row = int(indexes[0][i])
        col = int(indexes[1][i])
        prob = float(probability_matrix[row, col] * 100.0)
        top_results_array.append((row, col, prob))
    
    return top_results_array


def get_home_chance_of_x_goals(x_goals: int, probability_matrix: np.ndarray) -> float:
    """Calculate the probability of the home team scoring exactly X goals."""
    if x_goals < 0 or x_goals > 5:
        return 0.0
    return float(np.sum(probability_matrix[x_goals, :]))


def get_home_chance_of_x_goals_or_more(x_goals: int, probability_matrix: np.ndarray) -> float:
    """Calculate the probability of the home team scoring X or more goals."""
    if x_goals < 0 or x_goals > 5:
        return 0.0
    return float(np.sum(probability_matrix[x_goals:, :]))


def get_home_chance_of_more_than_x_goals(x_goals: int, probability_matrix: np.ndarray) -> float:
    """Calculate the probability of the home team scoring strictly more than X goals."""
    if x_goals < 0 or x_goals >= 5:
        return 0.0
    return float(np.sum(probability_matrix[x_goals + 1:, :]))


def get_home_chance_of_less_than_x_goals(x_goals: int, probability_matrix: np.ndarray) -> float:
    """Calculate the probability of the home team scoring strictly less than X goals."""
    if x_goals <= 0 or x_goals > 5:
        return 0.0
    return float(np.sum(probability_matrix[:x_goals, :]))


def get_away_chance_of_x_goals(x_goals: int, probability_matrix: np.ndarray) -> float:
    """Calculate the probability of the away team scoring exactly X goals."""
    if x_goals < 0 or x_goals > 5:
        return 0.0
    return float(np.sum(probability_matrix[:, x_goals]))


def get_away_chance_of_x_goals_or_more(x_goals: int, probability_matrix: np.ndarray) -> float:
    """Calculate the probability of the away team scoring X or more goals."""
    if x_goals < 0 or x_goals > 5:
        return 0.0
    return float(np.sum(probability_matrix[:, x_goals:]))


def get_away_chance_of_more_than_x_goals(x_goals: int, probability_matrix: np.ndarray) -> float:
    """Calculate the probability of the away team scoring strictly more than X goals."""
    if x_goals < 0 or x_goals >= 5:
        return 0.0
    return float(np.sum(probability_matrix[:, x_goals + 1:]))


def get_away_chance_of_less_than_x_goals(x_goals: int, probability_matrix: np.ndarray) -> float:
    """Calculate the probability of the away team scoring strictly less than X goals."""
    if x_goals <= 0 or x_goals > 5:
        return 0.0
    return float(np.sum(probability_matrix[:, :x_goals]))


def get_total_chance_of_more_than_x_goals(x_goals: int, probability_matrix: np.ndarray) -> float:
    """Calculate the combined probability of both teams scoring more than X total goals."""
    if x_goals < 0 or x_goals >= 10:
        return 0.0
    total_chance = 0.0
    for i in range(probability_matrix.shape[0]):
        for j in range(probability_matrix.shape[1]):
            if (i + j) > x_goals:
                total_chance += probability_matrix[i, j]
    return float(total_chance)


def get_total_chance_of_less_than_x_goals(x_goals: int, probability_matrix: np.ndarray) -> float:
    """Calculate the combined probability of both teams scoring less than X total goals."""
    if x_goals <= 0 or x_goals > 11:
        return 0.0
    total_chance = 0.0
    for i in range(probability_matrix.shape[0]):
        for j in range(probability_matrix.shape[1]):
            if (i + j) < x_goals:
                total_chance += probability_matrix[i, j]
    return float(total_chance)
