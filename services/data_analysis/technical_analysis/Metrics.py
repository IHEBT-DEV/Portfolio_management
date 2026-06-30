import logging
from typing import Dict, Any, Union
import pandas as pd

logger = logging.getLogger(__name__)


class Metrics:
    """
    High-performance technical analysis indicator engine.
    Computes optimized math overlays (SMA, EMA, Bollinger Bands, ATR, and Keltner Channels).
    """

    def __init__(self, ohlc_data: Union[pd.DataFrame, list, Dict[str, Any]]):
        """
        Initializes the Technical Indicator Engine workspace.
        Standardizes input structures into an insulated, working DataFrame.
        """
        # Ensure raw inputs are cleanly mapped into a structured pandas matrix frame
        if isinstance(ohlc_data, pd.DataFrame):
            self._master_df = ohlc_data.copy()
        else:
            self._master_df = pd.DataFrame(ohlc_data)

        # Verify baseline schema columns exist to prevent runtime indexing errors
        self._validate_schema()

    def _validate_schema(self) -> None:
        mandatory_fields = ['open', 'high', 'low', 'close']
        for field in mandatory_fields:
            if field not in self._master_df.columns:
                logger.warning(f"Technical Analysis Ingestion Warning: Missing expected field column [{field}].")

    def sma(self, window: int) -> pd.DataFrame:
        """
        Computes the Simple Moving Average (SMA) over a defined lookback series window.
        """
        df_output = self._master_df.copy()
        try:
            df_output[f'{window}_sma'] = df_output['close'].rolling(window=int(window), min_periods=1).mean()
        except Exception as error:
            logger.error(f"Failed to calculate SMA metric window [{window}]: {error}")
        return df_output

    def ema(self, window: int) -> pd.DataFrame:
        """
        Computes the Exponential Moving Average (EMA) utilizing exponential smoothing weights.
        """
        df_output = self._master_df.copy()
        try:
            df_output[f'{window}_ema'] = df_output['close'].ewm(span=int(window), adjust=False).mean()
        except Exception as error:
            logger.error(f"Failed to calculate EMA metric window [{window}]: {error}")
        return df_output

    def bollinger_bands(self, window: int, num_std_dev: float = 2.0) -> pd.DataFrame:
        """
        Computes Bollinger Bands volatility bands (Upper, Middle, and Lower envelopes).
        """
        df_output = self._master_df.copy()
        try:
            rolling_window = int(window)
            df_output['middleBand'] = df_output['close'].rolling(window=rolling_window, min_periods=1).mean()

            # Compute statistical standard deviation bounds
            rolling_std = df_output['close'].rolling(window=rolling_window, min_periods=1).std().fillna(0.0)

            df_output['upperBand'] = df_output['middleBand'] + (num_std_dev * rolling_std)
            df_output['lowerBand'] = df_output['middleBand'] - (num_std_dev * rolling_std)
        except Exception as error:
            logger.error(f"Failed to calculate Bollinger Bands over window [{window}]: {error}")
        return df_output

    def atr(self, window: int) -> pd.DataFrame:
        """
        Computes the Average True Range (ATR) measuring absolute baseline market asset volatility.
        """
        df_output = self._master_df.copy()
        try:
            # True Range = Max [(High - Low), abs(High - Close_prev), abs(Low - Close_prev)]
            high_low = df_output['high'] - df_output['low']
            high_close_prev = (df_output['high'] - df_output['close'].shift(1)).abs()
            low_close_prev = (df_output['low'] - df_output['close'].shift(1)).abs()

            df_output['TR'] = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            df_output['ATR'] = df_output['TR'].rolling(window=int(window), min_periods=1).mean()
        except Exception as error:
            logger.error(f"Failed to calculate ATR metric over window [{window}]: {error}")
        return df_output

    def keltner_channels(self, window: int, multiplier: float = 2.0) -> pd.DataFrame:
        """
        Computes Keltner Channels envelopes mapped against exponential metrics and smoothed ATR bounds.
        """
        try:
            # 1. Generate an isolated input frame for ATR metrics to eliminate structural key deletions
            atr_df = self.atr(window)
            df_output = self._master_df.copy()

            # 2. Extract calculations safely from our decoupled tracking frames
            df_output['Middle Channel'] = df_output['close'].rolling(window=int(window), min_periods=1).mean()
            df_output['Upper Channel'] = df_output['Middle Channel'] + (multiplier * atr_df['ATR'])
            df_output['Lower Channel'] = df_output['Middle Channel'] - (multiplier * atr_df['ATR'])

            return df_output
        except Exception as error:
            logger.error(f"Failed to calculate Keltner Channels over window [{window}]: {error}")
            return self._master_df.copy()
