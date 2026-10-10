"""Autonomous TomTom Traffic Flow collector with persistent daily quota tracking."""

import os
import signal
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from jro.exceptions import CongestionBudgetExceededError
from jro.logging_setup import logger

DEFAULT_DAILY_QUOTA = 2450  # buffer below hard 2500 limit


class TomTomBudgetTracker:
    """Persistent SQLite tracking for daily TomTom API usage."""

    def __init__(self, db_path: Path, daily_quota: int = DEFAULT_DAILY_QUOTA) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.daily_quota = daily_quota
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_usage (
                    date_utc TEXT PRIMARY KEY,
                    requests_used INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS congestion_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    point_id TEXT NOT NULL,
                    highway_class TEXT NOT NULL,
                    current_speed REAL,
                    free_flow_speed REAL,
                    speed_ratio REAL NOT NULL,
                    confidence REAL
                )
                """
            )
            conn.commit()

    def get_today_used(self) -> int:
        """Returns number of requests made today in UTC."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT requests_used FROM daily_usage WHERE date_utc = ?", (today,))
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def can_request(self) -> bool:
        """Checks whether daily budget remains available."""
        return self.get_today_used() < self.daily_quota

    def record_request(self) -> int:
        """Increments today's used request counter."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO daily_usage (date_utc, requests_used)
                VALUES (?, 1)
                ON CONFLICT(date_utc) DO UPDATE SET requests_used = requests_used + 1
                """,
                (today,),
            )
            conn.commit()
        return self.get_today_used()

    def record_observation(
        self,
        point_id: str,
        highway_class: str,
        current_speed: float,
        free_flow_speed: float,
        confidence: float,
    ) -> None:
        """Stores derived speed ratio aggregate (ToS compliant)."""
        now_str = datetime.now(UTC).isoformat()
        ratio = max(0.05, min(1.5, current_speed / max(1.0, free_flow_speed)))
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO congestion_observations (
                    timestamp_utc, point_id, highway_class,
                    current_speed, free_flow_speed, speed_ratio, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_str,
                    point_id,
                    highway_class,
                    current_speed,
                    free_flow_speed,
                    ratio,
                    confidence,
                ),
            )
            conn.commit()


class TomTomCongestionCollector:
    """Manages TomTom API calls, quota budgets, retries, and shutdown signals."""

    def __init__(
        self,
        api_key: str | None = None,
        db_path: Path | None = None,
        daily_quota: int = DEFAULT_DAILY_QUOTA,
    ) -> None:
        self.api_key = api_key or os.getenv("TOMTOM_API_KEY", "")
        db_file = db_path or Path("data/processed/congestion_budget.db")
        self.tracker = TomTomBudgetTracker(db_file, daily_quota=daily_quota)
        self.running = True
        self._setup_signals()

    def _setup_signals(self) -> None:
        def handler(signum: int, frame: Any) -> None:
            logger.info("Signal %d received: initiating graceful collector shutdown...", signum)
            self.running = False

        try:
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)
        except (ValueError, AttributeError):
            pass

    def fetch_flow_segment(
        self,
        point_id: str,
        lat: float,
        lon: float,
        highway_class: str,
    ) -> float | None:
        """Calls TomTom Flow Segment API for a coordinate and records derived ratio.

        Returns:
            Calculated speed ratio (current / free flow), or None if quota exhausted/skipped.
        """
        if not self.api_key:
            logger.warning("No TomTom API key configured; skipping live probe.")
            return None

        if not self.tracker.can_request():
            logger.warning(
                "Daily TomTom API quota reached (%d used). Stopping collection.",
                self.tracker.get_today_used(),
            )
            raise CongestionBudgetExceededError(
                f"Daily quota of {self.tracker.daily_quota} requests reached."
            )

        url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json"
        params = {
            "key": self.api_key,
            "point": f"{lat:.5f},{lon:.5f}",
            "unit": "KMPH",
        }

        try:
            resp = requests.get(url, params=params, timeout=10.0)
            self.tracker.record_request()

            if resp.status_code == 429:
                logger.warning("Received HTTP 429 Rate Limit from TomTom; sleeping 5 seconds...")
                time.sleep(5.0)
                return None

            if resp.status_code != 200:
                logger.warning("TomTom API error %d: %s", resp.status_code, resp.text[:200])
                return None

            data = resp.json()
            flow = data.get("flowSegmentData", {})
            curr_spd = float(flow.get("currentSpeed", 30.0))
            free_spd = float(flow.get("freeFlowSpeed", 40.0))
            conf = float(flow.get("confidence", 0.8))

            self.tracker.record_observation(
                point_id=point_id,
                highway_class=highway_class,
                current_speed=curr_spd,
                free_flow_speed=free_spd,
                confidence=conf,
            )
            ratio = curr_spd / max(1.0, free_spd)
            logger.info(
                "Probe %s: %s km/h / %s km/h (ratio=%.2f, used=%d)",
                point_id,
                curr_spd,
                free_spd,
                ratio,
                self.tracker.get_today_used(),
            )
            return ratio

        except requests.RequestException as e:
            logger.warning("Network failure during TomTom probe for %s: %s", point_id, e)
            return None
