import json
import logging
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)


class AnalyticsHelpers:
    """
    Advanced statistical evaluation and quantitative analysis helpers core.
    Executes cross-asset ordinary least squares (OLS) models and structural metrics distribution parsing.
    """

    def __init__(self, df: pd.DataFrame, df2: Optional[pd.DataFrame] = None):
        """
        Initializes the data tracking frames.
        """
        self.df = df.copy()
        self.df2 = df2.copy() if df2 is not None else None

    def standardize_asset(self, attribute_mean: float, attribute_std: float, attribute: str) -> pd.DataFrame:
        """
        Standardizes an asset attribute vector in-place.
        """
        if attribute_std <= 0:
            logger.warning(
                f"Standardization boundary alert: Standard deviation is non-positive for [{attribute}]. Skipping.")
            return self.df

        self.df[f"{attribute}_standardised"] = (self.df[attribute] - attribute_mean) / attribute_std
        return self.df

    def z_score(self, attribute_mean: float, attribute_std: float, attribute: str) -> pd.DataFrame:
        """
        Computes structural statistical Z-Scores cleanly for a target column timeline.
        """
        if attribute_std <= 0:
            logger.warning(f"Z-Score calculation boundary alert: Standard deviation limit collapsed for [{attribute}].")
            return self.df

        self.df['z_score'] = (self.df[attribute] - attribute_mean) / attribute_std
        return self.df

    def regression_model(self, attribute1: str, attribute2: str) -> Dict[str, Any]:
        """
        Computes an Ordinary Least Squares (OLS) cross-asset linear regression model.
        """
        if self.df2 is None or attribute2 not in self.df2.columns or attribute1 not in self.df.columns:
            logger.error("Regression modeling aborted: Mandatory dataframe reference or column mapping target absent.")
            return {}

        try:
            # Build clean data alignment map cleanly to prevent the drop dataframe execution bug
            df_regression = pd.DataFrame({
                'y': self.df[attribute1],
                'x': self.df2[attribute2]
            }).dropna()

            if len(df_regression) < 5:
                logger.warning(
                    "Insufficient data points across vector rows to formulate a valid OLS regression matrix.")
                return {}

            # Execute OLS using formula notation strings safely
            model = smf.ols(formula="y ~ x", data=df_regression).fit()

            return {
                'error_std': float(round(model.resid.std(), 4)),
                'beta': float(round(model.params.iloc[1], 4)),
                'alpha': float(round(model.params.iloc[0], 4)),
                'r-squared': float(round(model.rsquared, 4)),
                'min': float(self.df2[attribute2].min()),
                'max': float(self.df2[attribute2].max())
            }
        except Exception as error:
            logger.error(f"Mathematical algebraic exception inside OLS regression pipeline: {error}")
            return {}

    def spread_distribution(self, attribute: str) -> Dict[str, Any]:
        """
        Computes optimized histogram probabilities distribution matrix maps over a spread array timeline.
        """
        if attribute not in self.df.columns:
            logger.error(f"Histogram breakdown aborted: Specified mapping column target [{attribute}] absent.")
            return {}

        try:
            min_value = float(self.df[attribute].min())
            max_value = float(self.df[attribute].max())

            # Fallback handling to verify whether structural 'spread' key mapping layout is active
            target_col = 'spread' if 'spread' in self.df.columns else attribute
            list_spread = self.df[target_col].dropna().tolist()

            if not list_spread:
                return {}

            hist, bin_edges = np.histogram(list_spread, bins=20, range=(min_value, max_value))

            # Prevent dividing-by-zero on empty array configurations safely
            total_hits = hist.sum()
            unit_density = hist / total_hits if total_hits > 0 else np.zeros_like(hist)

            return {
                'title': f'{attribute} distribution',
                'x': {'title': attribute, 'values': [float(v) for v in bin_edges]},
                'y': {'title': 'probability', 'values': [float(p) for p in unit_density]}
            }
        except Exception as error:
            logger.error(f"Matrix array mathematical breakdown exception inside spread histogram tracking: {error}")
            return {}

    def correlation_history(self, asset_1_attribute: str, asset_2_attribute: str) -> Tuple[
        List[Dict[str, Any]], Dict[str, Any]]:
        """
        Computes live structural, rolling correlation history maps between dual pricing matrix files.
        """
        if self.df2 is None or 'date' not in self.df.columns or len(self.df) < 2:
            return [], {}

        try:
            df_corr = pd.DataFrame({'date': self.df['date']})

            # Enforce mathematical validation safety margin cutoff overrides over zero-second delays
            delay = self.df['date'].iloc[1] - self.df['date'].iloc[0]
            total_seconds = max(delay.total_seconds(), 1.0)
            hours = int(total_seconds / 3600)

            # Determine adaptive rolling calculation horizons cleanly
            if hours <= 1:
                rolling_window = 24
                roller_label = '1 day'
            elif hours >= 12:
                rolling_window = int((24 / max(hours, 1)) * 30)
                roller_label = '1 month'
            else:
                rolling_window = int((24 / max(hours, 1)) * 7)
                roller_label = '1 week'

            # Execute rolling correlation tracking directly
            df_corr['correlation'] = self.df[asset_1_attribute].rolling(int(max(rolling_window, 2))).corr(
                self.df2[asset_2_attribute])
            df_corr = df_corr.dropna()

            if df_corr.empty:
                return [], {}

            list_corr = df_corr['correlation'].tolist()
            hist, bin_edges = np.histogram(list_corr, bins='auto')

            total_hits = hist.sum()
            unit_density = hist / total_hits if total_hits > 0 else np.zeros_like(hist)

            hist_corr = {
                'title': f"{roller_label} rolled correlation",
                'x': {'title': 'spread', 'values': [float(v) for v in bin_edges]},
                'y': {'title': 'probability', 'values': [float(p) for p in unit_density]}
            }

            # High-performance native pandas record parsing optimization
            parsed_records = df_corr.to_dict(orient="records")
            # Cleanly format dates inside JSON serialization records
            for record in parsed_records:
                if isinstance(record.get('date'), pd.Timestamp):
                    record['date'] = record['date'].strftime('%Y-%m-%d %H:%M:%S')

            return parsed_records, hist_corr

        except Exception as error:
            logger.error(f"Operational anomaly encountered inside rolling correlation data channels: {error}")
            return [], {}

    def stats_resume(self, attribute: str) -> Dict[str, float]:
        """
        Generates advanced historical statistical descriptive profiles over an asset column layer.
        """
        if attribute not in self.df.columns or self.df[attribute].dropna().empty:
            return {}

        try:
            series = self.df[attribute].dropna()

            # Isolated vector calculation pattern cleanly strips out standard optimization delays
            mean_val = float(round(series.mean(), 4))
            std_val = float(round(series.std(), 4))
            skew_val = float(round(series.skew(), 4))
            kurt_val = float(round(series.kurt(), 4))
            median_val = float(round(series.median(), 4))

            # Fisher excess kurtosis calculation adjustment checks
            excess_kurtosis = float(round(kurt_val - 3.0, 4))

            return {
                'mean': mean_val,
                'std': std_val,
                'skewness': skew_val,
                'kurtosis': kurt_val,
                'excess_kurtosis': excess_kurtosis,
                'median': median_val
            }
        except Exception as error:
            logger.error(f"Descriptive data metrics compilation exception caught over [{attribute}]: {error}")
            return {}
