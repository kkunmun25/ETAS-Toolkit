import pytest
from unittest.mock import MagicMock, patch

from eq_toolkit.sources.registry import fetch_region


def test_california_registry():

    mock_get_events = MagicMock(return_value=[])

    with patch.dict(
        "eq_toolkit.sources.registry.SOURCES",
        {"comcat": mock_get_events},
    ):

        result = fetch_region(
            region="california",
            start_year=2010,
            end_year=2020,
            min_mag=2.5,
        )

    mock_get_events.assert_called_once_with(
        starttime="2010-01-01",
        endtime="2020-12-31",
        minmagnitude=2.5,
        minlatitude=32.0,
        maxlatitude=42.5,
        minlongitude=-125.0,
        maxlongitude=-113.0,
    )

    assert result == []


def test_unknown_region():

    with pytest.raises(ValueError):

        fetch_region(
            region="unknown",
            start_year=2010,
            end_year=2020,
            min_mag=2.5,
        )