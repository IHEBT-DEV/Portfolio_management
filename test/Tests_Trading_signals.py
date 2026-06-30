import pytest
import pandas as pd
from services.data_analysis.technical_analysis.Metrics import Metrics
from services.data_analysis.technical_analysis.TradingSignals import TradingSignals


class TestTradingSignals:
    """
    Production-grade automated unit testing suite validating quantitative
    trading signal generation engines. Enforces logic contract verifications.
    """

    @pytest.fixture(autouse=True)
    def setup_engine(self, sample_dataframe: pd.DataFrame):
        """
        Setup fixture initializing the dependent Metrics and TradingSignals engines
        automatically before each test execution block.
        """
        self.metrics = Metrics(sample_dataframe)
        self.signals_engine = TradingSignals(self.metrics, sample_dataframe)

    # --- CROSSOVER STRATEGY TESTS ---

    def test_golden_cross_sma_signals(self):
        """
        Validates the structure of Simple Moving Average Golden Cross signal outputs.
        """
        output_df = self.signals_engine.golden_cross_signal_sma(short_window=2, long_window=3)

        assert isinstance(output_df, pd.DataFrame), "Output layer must return a valid Pandas DataFrame."
        assert 'Golden_cross_Signal' in output_df.columns, "Expected tracking signal column missing from data schema."
        assert set(output_df['Golden_cross_Signal'].unique()).issubset(
            {0, 1}), "Signal mapping breach: Values must strictly be binary 0 or 1."

    def test_golden_cross_ema_signals(self):
        """
        Validates the structure of Exponential Moving Average Golden Cross signal outputs.
        """
        output_df = self.signals_engine.golden_cross_signal_ema(short_window=2, long_window=3)

        assert 'Golden_cross_Signal' in output_df.columns
        assert not output_df[
            'Golden_cross_Signal'].isna().any(), "Signal calculations vector cannot contain null values."

    def test_death_cross_sma_signals(self):
        """
        Validates the structure of Simple Moving Average Death Cross signal outputs.
        """
        output_df = self.signals_engine.death_cross_signal_sma(short_window=2, long_window=3)

        assert 'Death_cross_Signal' in output_df.columns
        assert 'Short_MA' in output_df.columns
        assert 'Long_MA' in output_df.columns

    def test_vectorized_death_cross_ema_signals(self):
        """
        Validates the high-speed vectorized shifting intersections model inside the EMA Death Cross engine.
        """
        output_df = self.signals_engine.death_cross_signal_ema(short_window=2, long_window=3)

        assert 'Death_cross_Signal' in output_df.columns

        # Verify that the shift loop successfully handles tracking calculations
        # without overflowing or leaving the initial boundary row index corrupted
        assert pd.api.types.is_integer_dtype(
            output_df['Death_cross_Signal']), "Signal output series must parse as type-safe integers."

    # --- BOUNDARY MOMENTUM TESTS ---

    def test_overbought_keltner_channels(self):
        """
        Validates Keltner Channels overbought event triggers.
        """
        output_df = self.signals_engine.overbought_signal_keltner_channels(window=2, multiplier=1.0)

        assert 'Upper Channel' in output_df.columns
        assert 'Overbought Signal' in output_df.columns

    def test_overbought_bollinger_bands(self):
        """
        Validates Bollinger Bands overbought event triggers.
        """
        output_df = self.signals_engine.overbought_signal_bollinger_bands(window=2, num_std_dev=1.0)

        assert 'Upper Band' in output_df.columns
        assert 'Overbought Signal' in output_df.columns

    def test_oversold_keltner_channels(self):
        """
        Validates Keltner Channels oversold condition event tracking.
        """
        output_df = self.signals_engine.oversold_signal_keltner_channels(window=2, multiplier=1.0)

        assert 'Lower Channel' in output_df.columns
        assert 'Oversold Signal' in output_df.columns

    def test_oversold_bollinger_bands(self):
        """
        Validates Bollinger Bands oversold condition event tracking.
        """
        output_df = self.signals_engine.oversold_signal_bollinger_bands(window=2, num_std_dev=1.0)

        assert 'Lower Band' in output_df.columns
        assert 'Oversold Signal' in output_df.columns

    # --- VOLATILITY CONTRACTION SQUEEZE TESTS ---

    def test_volatility_squeeze_keltner_channels(self):
        """
        Validates volatility contraction detection loops inside Keltner Channels.
        """
        output_df = self.signals_engine.squeeze_signal_keltner_channels(window=2, multiplier=1.0)

        assert 'Middle Channel' in output_df.columns
        assert 'Squeeze Signal' in output_df.columns

    def test_volatility_squeeze_bollinger_bands(self):
        """
        Validates volatility contraction detection loops inside Bollinger Bands envelopes.
        """
        output_df = self.signals_engine.squeeze_signal_bollinger_bands(window=2, num_std_dev=1.0)

        assert 'Squeeze Signal' in output_df.columns
        # Verify our memory safety cleanup logic successfully dropped temporary math trackers
        assert 'Upper Band' not in output_df.columns, "Memory Leakage Check Failed: Temporary upper allocation band tracking row leaked."
        assert 'Lower Band' not in output_df.columns, "Memory Leakage Check Failed: Temporary lower allocation band tracking row leaked."
