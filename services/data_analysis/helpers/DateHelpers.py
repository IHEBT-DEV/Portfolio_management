import logging
import re
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class DateHelpers:
    """
    High-performance chronological and timeline orchestration core.
    Parses complex multi-digit intervals and slices large date blocks into pagination-ready data chunks.
    """

    @staticmethod
    def parse_interval(interval: str) -> Tuple[int, str]:
        """
        Safely decomposes multi-digit parameter interval strings into numeric chunks and time frames.
        Examples: '15m' -> (15, 'm'), '1D' -> (1, 'd')
        """
        match = re.match(r"(\d+)([a-zA-Z])", interval.strip())
        if not match:
            logger.error(
                f"Linguistic interval configuration mismatch: Could not parse token [{interval}]. Defaulting to 1D.")
            return 1, 'd'

        coefficient_str, granularity_str = match.groups()
        return int(coefficient_str), granularity_str.lower()

    @staticmethod
    def number_of_days(timestamp1: Union[int, str], timestamp2: Union[int, str]) -> int:
        """
        Computes the absolute distance interval expressed in standard calendar days between two millisecond timestamps.
        """
        try:
            ts1 = int(timestamp1)
            ts2 = int(timestamp2)
            delta_days = int(abs(ts1 - ts2) / (1000 * 60 * 60 * 24))
            return max(delta_days, 1)  # Force floor boundaries of 1 day to prevent dividing-by-zero
        except (ValueError, TypeError) as casting_error:
            logger.error(f"Chronological conversion type tracking fault inside number_of_days: {casting_error}")
            return 1

    @classmethod
    def number_of_points(cls, start_date: Union[int, str], end_date: Union[int, str], interval: str) -> int:
        """
        Computes the expected count of target candle metrics slots across a specified timeline.
        """
        coefficient, granularity = cls.parse_interval(interval)
        nb_days = cls.number_of_days(start_date, end_date)

        # Map precise time-division coefficients based on structural time step increments
        if granularity == 's':
            nb_points = int(nb_days * 24 * 60 * 60 / coefficient)
        elif granularity == 'm':
            nb_points = int(nb_days * 24 * 60 / coefficient)
        elif granularity == 'h':
            nb_points = int(nb_days * 24 / coefficient)
        elif granularity == 'd':
            nb_points = int(nb_days / coefficient)
        elif granularity == 'w':
            nb_points = int(nb_days / (7 * coefficient))
        else:  # Monthly 'M' configurations defaults
            nb_points = int(nb_days / (30 * coefficient))

        return max(nb_points, 1)

    @classmethod
    def list_start_end_chunks(cls, start_date: Union[int, str], end_date: Union[int, str], interval: str) -> List[
        Dict[str, Any]]:
        """
        Slices broad chronological timelines into highly performant, 1,000-row batch execution lists.
        Guarantees flawless compatibility with API payload limits.
        """
        try:
            start_ts = int(start_date)
            end_ts = int(end_date)
        except (ValueError, TypeError):
            logger.error(
                "Data tracking bounds error: Start or end timestamp format corrupt. Aborting chunk generation.")
            return []

        nb_points = cls.number_of_points(start_ts, end_ts, interval)
        coefficient, granularity = cls.parse_interval(interval)

        # Scenario A: Payload data array securely sits within standard single-trip constraints
        if nb_points <= 1000:
            return [{'start_date': start_ts, 'limit': nb_points}]

        # Scenario B: Target dataset exceeds 1,000 parameters. Splitting timeline into waves
        list_starts_end_dates: List[Dict[str, Any]] = []

        # Compute exact step delay multipliers mapped in pure milliseconds units
        if granularity == 's':
            ms_delay = 1000 * coefficient
        elif granularity == 'm':
            ms_delay = 1000 * 60 * coefficient
        elif granularity == 'h':
            ms_delay = 1000 * 60 * 60 * coefficient
        elif granularity == 'd':
            ms_delay = 1000 * 60 * 60 * 24 * coefficient
        elif granularity == 'w':
            ms_delay = 1000 * 60 * 60 * 24 * 7 * coefficient
        else:  # Monthly interval mapping logic standard definitions
            ms_delay = 1000 * 60 * 60 * 24 * 30 * coefficient

        total_loops = (nb_points // 1000) + 1
        for i in range(total_loops):
            remaining_points = nb_points - (i * 1000)
            if remaining_points <= 0:
                break

            limit = 1000 if remaining_points >= 1000 else remaining_points
            computed_start = start_ts + (i * 1000 * ms_delay)

            # Boundary guard check to stop array appending if calculations step beyond target end timeline limit
            if computed_start >= end_ts:
                break

            list_starts_end_dates.append({
                'start_date': computed_start,
                'limit': limit
            })

        return list_starts_end_dates
