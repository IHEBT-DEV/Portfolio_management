from enum import Enum
from typing import List


class BinanceInterval(str, Enum):
    """
    Type-safe, immutable time intervals matching Binance API parameter mappings.
    Inherits from str to allow seamless json serialization and string comparisons.
    """
    ONE_SECOND = '1s'
    ONE_MINUTE = '1m'
    THREE_MINUTES = '3m'
    FIVE_MINUTES = '5m'
    FIFTEEN_MINUTES = '15m'
    THIRTY_MINUTES = '30m'
    ONE_HOUR = '1h'
    TWO_HOURS = '2h'
    FOUR_HOURS = '4h'
    SIX_HOURS = '6h'
    EIGHT_HOURS = '8h'
    TWELVE_HOURS = '12h'
    ONE_DAY = '1d'
    THREE_DAYS = '3d'
    ONE_WEEK = '1w'
    ONE_MONTH = '1M'

    @property
    def in_seconds(self) -> int:
        """
        Returns the mathematical equivalent of the interval configuration in seconds.
        Crucial for time-delta validation boundaries and rolling metrics loops.
        """
        # Parse the string token structure
        value_string = self.value
        unit = value_string[-1]
        amount = int(value_string[:-1])

        # Core time multiplier schema maps
        multipliers = {
            's': 1,
            'm': 60,
            'h': 3600,
            'd': 86400,
            'w': 604800,
            'M': 2592000  # Approximated standard financial month timeline (30 days)
        }

        return amount * multipliers.get(unit, 1)

    @classmethod
    def get_all_values(cls) -> List[str]:
        """
        Utility lookup helper to validate incoming raw external parameter packets.
        """
        return [item.value for item in cls]
