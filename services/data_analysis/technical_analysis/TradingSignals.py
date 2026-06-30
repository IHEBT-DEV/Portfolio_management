import logging
from typing import Dict, Any, Union
import pandas as pd
from services.data_analysis.technical_analysis.Metrics import Metrics

logger = logging.getLogger(__name__)


class TradingSignals:
    """
    Production-grade Quantitative Signal Generation Engine.
    Processes structural asset timelines to compute vectorized crossover, mean-reversion, and squeeze signals.
    """

    def __init__(self, metrics: Metrics, ohlc_data: Union[pd.DataFrame, list, Dict[str, Any]]):
        """
        Initializes the Trading Signals pipeline workspace.

        :param metrics: Pre-configured, high-performance Metrics service instance wrapper.
        :param ohlc_data: Core historical data inputs structured as a DataFrame or primitive sequence map.
        """
        self.metrics = metrics
        self._master_df = ohlc_data.copy() if isinstance(ohlc_data, pd.DataFrame) else pd.DataFrame(ohlc_data)

    def _prepare_base_data(self) -> pd.DataFrame:
        """
        Returns a fresh working copy of the base timeline matrix to protect source data integrity.
        """
        return self._master_df.copy()

    # --- GOLDEN CROSS SIGNAL ENGINE ---

    def golden_cross_signal_sma(self, short_window: int = 50, long_window: int = 200) -> pd.DataFrame:
        """
        Computes Golden Cross triggers utilizing Simple Moving Averages (SMA).
        Flags structural bullish regime entries where Fast SMA crosses above Slow SMA.
        """
        data = self._prepare_base_data()
        try:
            data['Short_MA'] = self.metrics.sma(short_window)[f'{short_window}_sma']
            data['Long_MA'] = self.metrics.sma(long_window)[f'{long_window}_sma']

            data['Golden_cross_Signal'] = 0
            # Vectorized logical assignment: Trigger 1 when the condition evaluates to true
            data.loc[data['Short_MA'] > data['Long_MA'], 'Golden_cross_Signal'] = 1
        except Exception as error:
            logger.error(f"Failed executing SMA Golden Cross signal matrix: {error}")
        return data

    def golden_cross_signal_ema(self, short_window: int = 50, long_window: int = 200) -> pd.DataFrame:
        """
        Computes Golden Cross triggers utilizing Exponential Moving Averages (EMA) for faster transition detection.
        """
        data = self._prepare_base_data()
        try:
            data['Short_MA'] = self.metrics.ema(short_window)[f'{short_window}_ema']
            data['Long_MA'] = self.metrics.ema(long_window)[f'{long_window}_ema']

            data['Golden_cross_Signal'] = 0
            data.loc[data['Short_MA'] > data['Long_MA'], 'Golden_cross_Signal'] = 1
        except Exception as error:
            logger.error(f"Failed executing EMA Golden Cross signal matrix: {error}")
        return data

    # --- DEATH CROSS SIGNAL ENGINE ---

    def death_cross_signal_sma(self, short_window: int = 50, long_window: int = 200) -> pd.DataFrame:
        """
        Computes Death Cross triggers utilizing Simple Moving Averages (SMA).
        Flags structural bearish breakdowns where Fast SMA drops below Slow SMA.
        """
        data = self._prepare_base_data()
        try:
            data['Short_MA'] = self.metrics.sma(short_window)[f'{short_window}_sma']
            data['Long_MA'] = self.metrics.sma(long_window)[f'{long_window}_sma']

            data['Death_cross_Signal'] = 0
            data.loc[data['Short_MA'] < data['Long_MA'], 'Death_cross_Signal'] = 1
        except Exception as error:
            logger.error(f"Failed executing SMA Death Cross signal matrix: {error}")
        return data

    def death_cross_signal_ema(self, short_window: int = 50, long_window: int = 200) -> pd.DataFrame:
        """
        Computes Death Cross triggers utilizing Exponential Moving Averages (EMA).
        Replaces manual row-by-row iteration loops with an instantaneous vectorized shift model.
        """
        data = self._prepare_base_data()
        try:
            data['Short_MA'] = self.metrics.ema(short_window)[f'{short_window}_ema']
            data['Long_MA'] = self.metrics.ema(long_window)[f'{long_window}_ema']

            # Vectorized Crossover Logic: Identifies the exact row step intersection crossing point
            # Condition: (Current Short < Current Long) AND (Previous Short >= Previous Long)
            is_below_now = data['Short_MA'] < data['Long_MA']
            was_above_prev = data['Short_MA'].shift(1) >= data['Long_MA'].shift(1)

            data['Death_cross_Signal'] = 0
            data.loc[is_below_now & was_above_prev, 'Death_cross_Signal'] = 1

        except Exception as error:
            logger.error(f"Failed executing Vectorized EMA Death Cross signal matrix: {error}")
        return data

    # --- OVERBOUGHT MOMENTUM CHANNELS ---

    def overbought_signal_keltner_channels(self, window: int = 20, multiplier: float = 2.0) -> pd.DataFrame:
        """
        Flags overbought strategy milestones when price breaks above the Upper Keltner Channel envelope.
        """
        data = self._prepare_base_data()
        try:
            data['Upper Channel'] = self.metrics.keltner_channels(window, multiplier)['Upper Channel']

            data['Overbought Signal'] = 0
            data.loc[data['close'] > data['Upper Channel'], 'Overbought Signal'] = 1
        except Exception as error:
            logger.error(f"Failed computing Keltner Overbought parameters: {error}")
        return data

    def overbought_signal_bollinger_bands(self, window: int = 20, num_std_dev: float = 2.0) -> pd.DataFrame:
        """
        Flags overbought strategy milestones when price spikes past the Upper Bollinger Band deviation layer.
        """
        data = self._prepare_base_data()
        try:
            data['Upper Band'] = self.metrics.bollinger_bands(window, num_std_dev)['upperBand']

            data['Overbought Signal'] = 0
            data.loc[data['close'] > data['Upper Band'], 'Overbought Signal'] = 1
        except Exception as error:
            logger.error(f"Failed computing Bollinger Overbought parameters: {error}")
        return data

    # --- OVERSOLD REVERSION BOUNDARIES ---

    def oversold_signal_keltner_channels(self, window: int = 20, multiplier: float = 2.0) -> pd.DataFrame:
        """
        Flags oversold strategy milestones when price drops below the Lower Keltner Channel envelope.
        """
        data = self._prepare_base_data()
        try:
            data['Lower Channel'] = self.metrics.keltner_channels(window, multiplier)['Lower Channel']

            data['Oversold Signal'] = 0
            data.loc[data['close'] < data['Lower Channel'], 'Oversold Signal'] = 1
        except Exception as error:
            logger.error(f"Failed computing Keltner Oversold parameters: {error}")
        return data

    def oversold_signal_bollinger_bands(self, window: int = 20, num_std_dev: float = 2.0) -> pd.DataFrame:
        """
        Flags oversold strategy milestones when price drops below the Lower Bollinger Band standard deviation.
        """
        data = self._prepare_base_data()
        try:
            data['Lower Band'] = self.metrics.bollinger_bands(window, num_std_dev)['lowerBand']

            data['Oversold Signal'] = 0
            data.loc[data['close'] < data['Lower Band'], 'Oversold Signal'] = 1
        except Exception as error:
            logger.error(f"Failed computing Bollinger Oversold parameters: {error}")
        return data

    # --- VOLATILITY SQUEEZE STRATEGIES ---

    def squeeze_signal_keltner_channels(self, window: int = 20, multiplier: float = 2.0) -> pd.DataFrame:
        """
        Detects volatility squeeze contraction periods within Keltner Channel envelopes.
        """
        data = self._prepare_base_data()
        try:
            keltner_channels = self.metrics.keltner_channels(window, multiplier)
            data['Middle Channel'] = keltner_channels['Middle Channel']
            data['Upper Channel'] = keltner_channels['Upper Channel']
            data['Lower Channel'] = keltner_channels['Lower Channel']

            channel_width = data['Upper Channel'] - data['Lower Channel']
            rolling_max_width = channel_width.rolling(window=int(window)).max()

            data['Squeeze Signal'] = 0
            data.loc[channel_width == rolling_max_width, 'Squeeze Signal'] = 1
        except Exception as error:
            logger.error(f"Failed computing Keltner Squeeze signals: {error}")
        return data

    def squeeze_signal_bollinger_bands(self, window: int = 20, num_std_dev: float = 2.0) -> pd.DataFrame:
        """
        Detects volatility squeeze contraction periods within Bollinger Band bands.
        """
        data = self._prepare_base_data()
        try:
            bollinger_bands = self.metrics.bollinger_bands(window, num_std_dev)
            data['Upper Band'] = bollinger_bands['upperBand']
            data['Lower Band'] = bollinger_bands['lowerBand']

            band_width = data['Upper Band'] - data['Lower Band']
            rolling_max_width = band_width.rolling(window=int(window)).max()

            data['Squeeze Signal'] = 0
            data.loc[band_width == rolling_max_width, 'Squeeze Signal'] = 1

            # Cleanly drop temporary calculation columns to save memory footprint weight
            data.drop(['Upper Band', 'Lower Band'], axis=1, inplace=True, errors='ignore')
        except Exception as error:
            logger.error(f"Failed computing Bollinger Squeeze signals: {error}")
        return data
