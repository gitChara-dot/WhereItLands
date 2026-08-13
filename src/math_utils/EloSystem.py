import numpy as np
from typing import Dict


class EloSystem:
    """Object-oriented system to track and calculate team Elo ratings using primitive types."""

    def __init__(self, k_values: Dict[str, int], initial_elo: float = 1500.0) -> None:
        self.k_values: Dict[str, int] = k_values
        self.initial_elo: float = initial_elo
        self.team_elo: Dict[str, float] = {}

    def get_elo(self, team: str) -> float:
        """Return the current Elo rating for a team, initializing it if absent."""
        if team not in self.team_elo:
            self.team_elo[team] = self.initial_elo
        return self.team_elo[team]

    def set_elo(self, team: str, elo: float) -> None:
        """Manually set the Elo rating of a team."""
        self.team_elo[team] = elo

    def get_weight(self, tournament: str) -> int:
        """Retrieve the K-factor weight for a given tournament, defaulting to 10."""
        return self.k_values.get(tournament, 10)

    @staticmethod
    def get_winner_result(team_goals: int, opponent_goals: int) -> float:
        """Calculate the match outcome score (1.0 for win, 0.5 for draw, 0.0 for loss)."""
        if team_goals > opponent_goals:
            return 1.0
        if team_goals == opponent_goals:
            return 0.5
        return 0.0

    @staticmethod
    def get_goal_multiplier(team_goals: int, opponent_goals: int) -> float:
        """Compute the goal difference multiplier to adjust Elo point swings."""
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
        """Calculate the expected match outcome using the logistic Elo formula with optional advantage."""
        adjusted_team_elo = team_elo + elo_advantage
        return 1.0 / (np.power(10.0, (-(adjusted_team_elo - opponent_elo) / 400.0)) + 1.0)

    def calculate_new_elo(
        self, 
        current_elo: float, 
        goal_multiplier: float, 
        K: int, 
        result: float, 
        expected_result: float
    ) -> float:
        """Compute the updated Elo rating value without mutating instance state."""
        return current_elo + (K * goal_multiplier) * (result - expected_result)

    def update_elo(
        self, 
        team: str, 
        goal_multiplier: float, 
        K: int, 
        result: float, 
        expected_result: float
    ) -> float:
        """Update and return the team Elo rating in the internal state dictionary."""
        current_elo = self.get_elo(team)
        new_elo = self.calculate_new_elo(current_elo, goal_multiplier, K, result, expected_result)
        self.set_elo(team, new_elo)
        return new_elo