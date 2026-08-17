"""Unit tests for data processing and pipeline transformations."""

import pytest
import pandas as pd
import numpy as np

from src.math_utils.EloSystem import EloSystem
from src.data_processing.pipeline import (
    filter_relevant_games,
    build_team_history,
    get_available_teams,
)


@pytest.fixture
def sample_match_dataframe() -> pd.DataFrame:
    """Fixture returning a mock dataframe of international matches."""
    data = {
        "date": pd.to_datetime(["2020-01-10", "2020-02-15", "2020-03-20"]),
        "home_team": ["Argentina", "France", "Brazil"],
        "away_team": ["Brazil", "Germany", "Argentina"],
        "home_score": [2, 1, 0],
        "away_score": [1, 1, 2],
        "home_elo": [1800.0, 1750.0, 1820.0],
        "away_elo": [1820.0, 1720.0, 1800.0],
        "tournament": ["FIFA World Cup", "Friendly", "Copa América"],
        "country": ["Argentina", "France", "Brazil"],
        "neutral": [False, False, False],
    }
    return pd.DataFrame(data)


def test_filter_relevant_games(sample_match_dataframe: pd.DataFrame) -> None:
    """Verify that competitive matches are retained and winner column is encoded correctly."""
    filtered = filter_relevant_games(sample_match_dataframe)
    # The friendly match should be filtered out based on date boundary
    assert len(filtered) == 2
    assert "winner" in filtered.columns
    # Argentina vs Brazil (2-1) should encode winner as 1
    assert filtered.iloc[0]["winner"] == 1


def test_get_available_teams() -> None:
    """Verify that get_available_teams returns a sorted unique list of team names."""
    df_history = pd.DataFrame({
        "team": ["Brazil", "Argentina", "France", "Argentina"]
    })
    teams = get_available_teams(df_history)
    assert teams == ["Argentina", "Brazil", "France"]
