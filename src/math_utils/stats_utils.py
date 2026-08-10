import numpy as np
from scipy.stats import poisson
from typing import List, Tuple
def get_coles_dixon_correction(home_goals: int, away_goals: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    """Aplica la corrección de Coles-Dixon para resultados bajos."""
    x = home_goals
    y = away_goals
    if x == 0 and y == 0:
        return 1 - lambda_home * lambda_away * rho
    if x == 0 and y == 1:
        return 1 + lambda_home * rho
    if x == 1 and y == 0:
        return 1 + lambda_away * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0

def process_match(lambda_home: float, lambda_away: float, rho: float) -> np.ndarray:
    """
    Obtiene la probabilidad para cada resultado i-j, para 0<=i<=5 y 0<=j<=5.
    Retorna una matriz de probabilidades de 6x6.
    """
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

def get_chances(probability_matrix: np.ndarray) -> list:
    """
    Obtiene las probabilidades de victoria local, empate y victoria visitante.
    """
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
    """Obtiene el resultado más probable como string 'i-j'."""
    index = np.unravel_index(np.argmax(probability_matrix), probability_matrix.shape)
    return f'{index[0]}-{index[1]}'

def get_likely_result_tuple(probability_matrix: np.ndarray) -> list:
    """Obtiene el resultado más probable como una lista [i, j]."""
    index = np.unravel_index(np.argmax(probability_matrix), probability_matrix.shape)
    return [int(index[0]), int(index[1])]

def get_top_x_probabilities(probability_matrix: np.ndarray, iterations: int) -> dict:
    """Obtiene los 'iterations' resultados más probables de un partido, como diccionario."""
    flattened_matrix = probability_matrix.flatten()
    if iterations <= 0 or iterations >= len(flattened_matrix):
        print("Invalid iteration number")
        return {}
    
    top_x = np.argsort(flattened_matrix)[-iterations:][::-1]
    indexes = np.unravel_index(top_x, probability_matrix.shape)

    top_x_probabilities_str = {}
    for i in range(iterations):
        row = indexes[0][i]
        col = indexes[1][i]
        prob = probability_matrix[row, col] * 100
        coord_str = f'{row}-{col}'
        top_x_probabilities_str[coord_str] = float(prob)
    
    return top_x_probabilities_str

def get_top_x_probabilities_array(probability_matrix: np.ndarray, iterations: int) -> List[Tuple[int, int, float]]:
    """Obtiene los 'iterations' resultados más probables de un partido, como array de 3 partes."""
    flattened_matrix = probability_matrix.flatten()
    if iterations <= 0 or iterations >= len(flattened_matrix):
        print("Invalid iteration number")
        return []
    
    top_x = np.argsort(flattened_matrix)[-iterations:][::-1]
    indexes = np.unravel_index(top_x, probability_matrix.shape)

    top_results_array = []


    for i in range(iterations):
        current_result_array = np.array([])
        row = indexes[0][i]
        col = indexes[1][i]
        prob = probability_matrix[row, col] * 100

        top_results_array.append((row, col, prob))
    
    return top_results_array
# -------------------------------------------------------------
# Funciones auxiliares para calcular probabilidades de N goles
# -------------------------------------------------------------

def get_home_chance_of_x_goals(x_goals: int, probability_matrix: np.ndarray) -> float:
    if x_goals < 0 or x_goals > 5:
        return 0.0
    return float(np.sum(probability_matrix[x_goals, :]))

def get_home_chance_of_x_goals_or_more(x_goals: int, probability_matrix: np.ndarray) -> float:
    if x_goals < 0 or x_goals > 5:
        return 0.0
    return float(np.sum(probability_matrix[x_goals:, :]))

def get_home_chance_of_more_than_x_goals(x_goals: int, probability_matrix: np.ndarray) -> float:
    if x_goals < 0 or x_goals >= 5:
        return 0.0
    return float(np.sum(probability_matrix[x_goals+1:, :]))

def get_home_chance_of_less_than_x_goals(x_goals: int, probability_matrix: np.ndarray) -> float:
    if x_goals <= 0 or x_goals > 5:
        return 0.0
    return float(np.sum(probability_matrix[:x_goals, :]))

def get_away_chance_of_x_goals(x_goals: int, probability_matrix: np.ndarray) -> float:
    if x_goals < 0 or x_goals > 5:
        return 0.0
    return float(np.sum(probability_matrix[:, x_goals]))

def get_away_chance_of_x_goals_or_more(x_goals: int, probability_matrix: np.ndarray) -> float:
    if x_goals < 0 or x_goals > 5:
        return 0.0
    return float(np.sum(probability_matrix[:, x_goals:]))

def get_away_chance_of_more_than_x_goals(x_goals: int, probability_matrix: np.ndarray) -> float:
    if x_goals < 0 or x_goals >= 5:
        return 0.0
    return float(np.sum(probability_matrix[:, x_goals+1:]))

def get_away_chance_of_less_than_x_goals(x_goals: int, probability_matrix: np.ndarray) -> float:
    if x_goals <= 0 or x_goals > 5:
        return 0.0
    return float(np.sum(probability_matrix[:, :x_goals]))

def get_total_chance_of_more_than_x_goals(x_goals: int, probability_matrix: np.ndarray) -> float:
    if x_goals < 0 or x_goals >= 10: # Sum can go up to 10
        return 0.0
    total_chance = 0.0
    for i in range(probability_matrix.shape[0]):
        for j in range(probability_matrix.shape[1]):
            if (i + j) > x_goals:
                total_chance += probability_matrix[i, j]
    return float(total_chance)

def get_total_chance_of_less_than_x_goals(x_goals: int, probability_matrix: np.ndarray) -> float:
    if x_goals <= 0 or x_goals > 11:
        return 0.0
    total_chance = 0.0
    for i in range(probability_matrix.shape[0]):
        for j in range(probability_matrix.shape[1]):
            if (i + j) < x_goals:
                total_chance += probability_matrix[i, j]
    return float(total_chance)
