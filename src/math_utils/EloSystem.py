import numpy as np
from typing import Dict, Optional

class EloSystem:
    """
    Clase orientada a objetos para manejar el sistema ELO de los equipos.
    No depende de pandas ni DataFrames, opera con tipos primitivos.
    """
    def __init__(self, k_values: Dict[str, int], initial_elo: float = 1500.0):
        self.k_values = k_values
        self.initial_elo = initial_elo
        self.team_elo: Dict[str, float] = {}

    def get_elo(self, team: str) -> float:
        """Devuelve el ELO actual de un equipo. Si no existe, lo inicializa con el valor por defecto."""
        if team not in self.team_elo:
            self.team_elo[team] = self.initial_elo
        return self.team_elo[team]

    def set_elo(self, team: str, elo: float) -> None:
        """Establece el ELO de un equipo manualmente."""
        self.team_elo[team] = elo

    def get_weight(self, tournament: str) -> int:
        """Obtiene el K-weight para un torneo específico. Por defecto es 10."""
        return self.k_values.get(tournament, 10)

    @staticmethod
    def get_winner_result(team_goals: int, opponent_goals: int) -> float:
        """Calcula el resultado para el equipo (1.0 = victoria, 0.5 = empate, 0.0 = derrota)."""
        if team_goals > opponent_goals:
            return 1.0
        if team_goals == opponent_goals:
            return 0.5
        return 0.0

    @staticmethod
    def get_goal_multiplier(team_goals: int, opponent_goals: int) -> float:
        """Calcula el multiplicador basado en la diferencia de goles."""
        diff = abs(team_goals - opponent_goals)
        if diff <= 1:
            return 1.0
        elif diff == 2:
            return 1.5
        elif diff == 3:
            return 1.75
        else:
            return 1.75 + (diff - 3) / 8.0

    @staticmethod
    def get_expected_result(team_elo: float, opponent_elo: float, elo_advantage: float = 0.0) -> float:
        """Calcula el resultado esperado basado en la diferencia de ELO, incluyendo posibles ventajas (localía)."""
        adjusted_team_elo = team_elo + elo_advantage
        return 1 / (np.power(10, (-(adjusted_team_elo - opponent_elo) / 400.0)) + 1)

    def calculate_new_elo(self, current_elo: float, goal_multiplier: float, K: int, result: float, expected_result: float) -> float:
        """Calcula cuál sería el nuevo ELO sin modificar el estado."""
        return current_elo + (K * goal_multiplier) * (result - expected_result)

    def update_elo(self, team: str, goal_multiplier: float, K: int, result: float, expected_result: float) -> float:
        """Calcula y actualiza el ELO del equipo en el estado."""
        current_elo = self.get_elo(team)
        new_elo = self.calculate_new_elo(current_elo, goal_multiplier, K, result, expected_result)
        self.set_elo(team, new_elo)
        return new_elo