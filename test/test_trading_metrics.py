import pytest
import pandas as pd
from services.data_analysis.technical_analysis.Metrics import Metrics


class TestTradingMetrics:
    """
    Production-grade automated unit testing suite validating quantitative
    technical analysis indicators. Enforces isolated in-memory assertions.
    """

    def test_sma_calculation(self, sample_dataframe: pd.DataFrame):
        """
        Validates that Simple Moving Average (SMA) columns are correctly generated
        and math outputs execute accurately over specified windows.
        """
        # Instantiate service cleanly utilizing our pre-aligned mock data fixture
        metrics_service = Metrics(sample_dataframe)
        window = 3

        output_df = metrics_service.sma(window=window)
        target_col = f'{window}_sma'

        # Strict Verification Assertions
        assert isinstance(output_df, pd.DataFrame), "Output layer must return a valid Pandas DataFrame object."
        assert target_col in output_df.columns, f"Expected output column tracking indicator label missing: [{target_col}]"
        assert not output_df[
            target_col].isna().all(), "Computed moving average series column cannot be completely empty."

        # Verify specific structural row mathematical output: index 2 should represent mean of index 0, 1, 2 close inputs
        expected_sma = (sample_dataframe['close'].iloc[0] + sample_dataframe['close'].iloc[1] +
                        sample_dataframe['close'].iloc[2]) / 3.0
        assert abs(output_df[target_col].iloc[
                       2] - expected_sma) < 1e-4, "Mathematical deviation boundary check failed on SMA output."

    def test_ema_calculation(self, sample_dataframe: pd.DataFrame):
        """
        Validates Exponential Moving Average (EMA) structural column generation boundaries.
        """
        metrics_service = Metrics(sample_dataframe)
        window = 3

        output_df = metrics_service.ema(window=window)
        target_col = f'{window}_ema'

        assert target_col in output_df.columns, f"Expected output column tracking indicator label missing: [{target_col}]"
        assert output_df[target_col].iloc[
                   -1] > 0, "Calculated asset exponential average metrics must resolve to positive bounds."

    def test_bollinger_bands_generation(self, sample_dataframe: pd.DataFrame):
        """
        Validates Bollinger Bands envelopes (Upper, Middle, Lower) are mapped out safely.
        """
        metrics_service = Metrics(sample_dataframe)
        window = 3
        num_std_dev = 2.0

        output_df = metrics_service.bollinger_bands(window=window, num_std_dev=num_std_dev)

        # Confirm absolute envelope hierarchy conditions remain true across all timeline rows: Lower <= Middle <= Upper
        assert 'middleBand' in output_df.columns
        assert 'upperBand' in output_df.columns
        assert 'lowerBand' in output_df.columns

        for idx in range(len(output_df)):
            assert output_df['lowerBand'].iloc[idx] <= output_df['middleBand'].iloc[
                idx], "Volatility Envelope Paradox: Lower band exceeds middle channel."
            assert output_df['middleBand'].iloc[idx] <= output_df['upperBand'].iloc[
                idx], "Volatility Envelope Paradox: Middle channel exceeds upper band."

    def test_atr_volatility_tracking(self, sample_dataframe: pd.DataFrame):
        """
        Validates Average True Range (ATR) high-low-close vector calculations.
        """
        metrics_service = Metrics(sample_dataframe)
        window = 3

        output_df = metrics_service.atr(window=window)

        assert 'TR' in output_df.columns, "Intermediate True Range indicator variable mapping column missing."
        assert 'ATR' in output_df.columns, "Primary Average True Range indicator metrics tracking column missing."
        assert (output_df['ATR'] >= 0).all(), "Volatility data arrays cannot resolve to negative absolute numbers."

    def test_keltner_channels_isolation(self, sample_dataframe: pd.DataFrame):
        """
        Validates Keltner Channels envelope matrices and checks for safe internal key cleanups.
        """
        metrics_service = Metrics(sample_dataframe)
        window = 3

        output_df = metrics_service.keltner_channels(window=window, multiplier=2.0)

        assert 'Middle Channel' in output_df.columns
        assert 'Upper Channel' in output_df.columns
        assert 'Lower Channel' in output_df.columns

        # Verify that temporary math columns 'TR' and 'ATR' are completely scrubbed to protect memory
        assert 'TR' not in output_df.columns, "Memory Leakage: Temporary calculation field [TR] leaked into output layer."
        assert 'ATR' not in output_df.columns, "Memory Leakage: Temporary calculation field [ATR] leaked into output layer."
