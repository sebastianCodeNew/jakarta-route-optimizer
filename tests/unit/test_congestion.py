"""Unit tests for TomTom API daily quota budgeting and tracker persistence."""

from jro.congestion.collector import TomTomBudgetTracker


def test_budget_tracker_daily_quota(tmp_path) -> None:
    """Verifies that budget tracker enforces daily threshold and records usage."""
    db_file = tmp_path / "budget_test.db"
    tracker = TomTomBudgetTracker(db_file, daily_quota=5)

    assert tracker.can_request()
    assert tracker.get_today_used() == 0

    # Record 5 requests
    for i in range(5):
        assert tracker.can_request()
        tracker.record_request()

    assert tracker.get_today_used() == 5
    # Limit reached!
    assert not tracker.can_request()

    # Re-instantiate from existing file (persistence check)
    tracker_reopened = TomTomBudgetTracker(db_file, daily_quota=5)
    assert tracker_reopened.get_today_used() == 5
    assert not tracker_reopened.can_request()


def test_observation_speed_ratio_bounds(tmp_path) -> None:
    """Tests observation logging and speed ratio clamping."""
    db_file = tmp_path / "budget_test.db"
    tracker = TomTomBudgetTracker(db_file, daily_quota=100)

    # Standard congested
    tracker.record_observation(
        "P01", "primary", current_speed=18.0, free_flow_speed=40.0, confidence=0.9
    )
    # Stalled
    tracker.record_observation(
        "P02", "primary", current_speed=0.5, free_flow_speed=40.0, confidence=0.8
    )

    import sqlite3

    with sqlite3.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute("SELECT speed_ratio FROM congestion_observations ORDER BY id ASC")
        rows = cur.fetchall()
        assert len(rows) == 2
        ratio1 = rows[0][0]
        ratio2 = rows[1][0]
        assert abs(ratio1 - (18.0 / 40.0)) < 1e-4
        assert ratio2 >= 0.05  # clamped minimum
