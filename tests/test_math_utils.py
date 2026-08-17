"""Unit tests for mathematical and statistical modeling utilities."""

import pytest
import numpy as np

from src.math_utils.EloSystem import EloSystem
from src.math_utils.stats_utils import (
    get_coles_dixon_correction,
    process_match,
    get_chances,
    get_top_x_probabilities_array,
)


class TestEloSystem:
    """Test suite for the dynamic EloSystem implementation."""

    def test_default_initial_elo(self) -> None:
        """Verify that unregistered teams receive the default initial rating."""
        elo = EloSystem(k_values={"FIFA World Cup": 60}, initial_elo=1500.0)
        assert elo.get_elo("Argentina") == 1500.0

    def test_expected_result_symmetry(self) -> None:
        """Verify that equal ratings yield an expected win probability of exactly 0.5."""
        elo = EloSystem(k_values={}, initial_elo=1500.0)
        expected = elo.get_expected_result(1500.0, 1500.0)
        assert np.isclose(expected, 0.5)

    def test_expected_result_higher_rating(self) -> None:
        """Verify that a higher rated team has an expected outcome greater than 0.5."""
        elo = EloSystem(k_values={}, initial_elo=1500.0)
        expected = elo.get_expected_result(1800.0, 1500.0)
        assert expected > 0.5

    def test_winner_result_outcomes(self) -> None:
        """Verify match outcome scores for win (1.0), draw (0.5), and loss (0.0)."""
        assert EloSystem.get_winner_result(3, 1) == 1.0
        assert EloSystem.get_winner_result(2, 2) == 0.5
        assert EloSystem.get_winner_result(0, 1) == 0.0

    def test_goal_multiplier_scaling(self) -> None:
        """Verify that margin-of-victory multipliers scale monotonically."""
        assert EloSystem.get_goal_multiplier(1, 0) == 1.0
        assert EloSystem.get_goal_multiplier(2, 0) == 1.5
        assert EloSystem.get_goal_multiplier(3, 0) == 1.75
        assert EloSystem.get_goal_multiplier(5, 0) > 1.75

    def test_update_elo_increases_on_win(self) -> None:
        """Verify that winning against an equal opponent strictly increases team rating."""
        elo = EloSystem(k_values={"FIFA World Cup": 60}, initial_elo=1500.0)
        new_rating = elo.update_elo(
            team="Argentina",
            goal_multiplier=1.0,
            K=60,
            result=1.0,
            expected_result=0.5,
        )
        assert new_rating > 1500.0
        assert elo.get_elo("Argentina") == new_rating


class TestStatsUtils:
    """Test suite for bivariate Poisson matrix and Coles-Dixon calculations."""

    def test_coles_dixon_correction_values(self) -> None:
        """Verify Coles-Dixon adjustment factors for low-scoring match outcomes."""
        rho = -0.05
        lh, la = 1.4, 1.1

        assert np.isclose(get_coles_dixon_correction(0, 0, lh, la, rho), 1.0 - lh * la * rho)
        assert np.isclose(get_coles_dixon_correction(0, 1, lh, la, rho), 1.0 + lh * rho)
        assert np.isclose(get_coles_dixon_correction(1, 0, lh, la, rho), 1.0 + la * rho)
        assert np.isclose(get_coles_dixon_correction(1, 1, lh, la, rho), 1.0 - rho)
        assert get_coles_dixon_correction(2, 1, lh, la, rho) == 1.0

    def test_process_match_matrix_sum(self) -> None:
        """Verify that the 6x6 score probability matrix integrates to approximately 1.0."""
        matrix = process_match(lambda_home=1.5, lambda_away=1.1, rho=-0.05)
        total_prob = float(matrix.sum())
        assert 0.95 <= total_prob <= 1.05

    def test_get_chances_partition_sum(self) -> None:
        """Verify that aggregate home win, draw, and away win sum to the total matrix mass."""
        matrix = process_match(lambda_home=1.8, lambda_away=0.9, rho=-0.05)
        chances = get_chances(matrix)
        assert len(chances) == 3
        assert np.isclose(sum(chances), matrix.sum(), atol=1e-3)
        assert chances[0] > chances[2]  # Higher home lambda implies higher home win probability

    def test_top_x_probabilities_array_ordering(self) -> None:
        """Verify that top-N scorelines are sorted in strictly descending likelihood."""
        matrix = process_match(lambda_home=1.4, lambda_away=1.0, rho=-0.05)
        top_3 = get_top_x_probabilities_array(matrix, iterations=3)

        assert len(top_3) == 3
        # Check descending order
        assert top_3[0][2] >= top_3[1][2] >= top_3[2][2]
        # Check percentage bounds
        for home_goals, away_goals, pct in top_3:
            assert 0 <= home_goals <= 5
            assert 0 <= away_goals <= 5
            assert 0.0 < pct < 100.0
