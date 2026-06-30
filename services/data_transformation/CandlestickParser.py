import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class CandlestickParser:
    """
    High-performance data transformation utility engine.
    Parses and standardizes raw external exchange payload data into normalized schemas.
    """

    @classmethod
    def parse_spot_by_market_currency(
            cls, data: List[List[Any]], market: str, currency: str
    ) -> List[Dict[str, Any]]:
        """
        Public gateway to route and execute market-specific data mapping protocols.
        """
        if not data:
            return []

        market_key = market.strip().upper()

        if market_key == "BINANCE":
            return cls._binance_ohlcv_parser(data, currency)

        logger.warning(f"Transformation route aborted: Unsupported exchange mapping requested [{market}].")
        return []

    @staticmethod
    def _binance_ohlcv_parser(data: List[List[Any]], currency: str) -> List[Dict[str, Any]]:
        """
        Transforms raw Binance OHLCV matrix lines into sanitized relational dictionary layers.
        Enforces defensive validation metrics over array bounds and value parsing.
        """
        parsed_records: List[Dict[str, Any]] = []

        for row_index, row in enumerate(data):
            # Enforce array length constraint validation (Binance OHLCV expects at least 6 primary nodes)
            if not isinstance(row, (list, tuple)) or len(row) < 6:
                logger.warning(
                    f"Skipping corrupted matrix sequence at line row index [{row_index}]. Matrix segment invalid.")
                continue

            try:
                # Convert UNIX millisecond timestamp to a timezone-aware UTC datetime layout
                timestamp_seconds = int(row[0]) / 1000.0
                utc_datetime = datetime.fromtimestamp(timestamp_seconds, tz=timezone.utc)

                parsed_records.append({
                    'currency': currency.upper(),
                    'date': utc_datetime,
                    'open': float(row[1]),
                    'high': float(row[2]),
                    'low': float(row[3]),
                    'close': float(row[4]),
                    'volume': float(row[5])
                })
            except (ValueError, TypeError) as casting_error:
                logger.error(
                    f"Casting anomaly encountered at record index [{row_index}]: {casting_error}. Skipping line segment."
                )
                continue

        return parsed_records
