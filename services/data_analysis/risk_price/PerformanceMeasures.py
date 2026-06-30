import logging
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceMeasures:
    """
    Advanced financial portfolio performance measures engine.
    Computes risk-adjusted performance overlays (Sharpe, Treynor, Jensen's Alpha, and SML).
    """

    @staticmethod
    def calculate_portfolio_beta(portfolio_weights: Dict[str, float], asset_risks: pd.DataFrame) -> float:
        """
        Calculates the aggregated Portfolio Beta via high-speed vectorized dot products.
        Expects asset_risks to be a DataFrame indexed by asset symbols containing a 'Beta_asset' column.
        """
        if not portfolio_weights or asset_risks.empty:
            return 0.0

        try:
            # Align keys and extract arrays cleanly
            assets = list(portfolio_weights.keys())
            weights_vector = np.array([portfolio_weights[asset] for asset in assets], dtype=float)

            # Extract corresponding betas safely from the indexed DataFrame
            betas_vector = asset_risks.loc[assets, 'Beta_asset'].to_numpy(dtype=float)

            # Matrix Dot Product: Sum(Weight_i * Beta_i) executed instantly in low-level C
            return float(np.dot(weights_vector, betas_vector))

        except KeyError as missing_key:
            logger.error(f"Beta aggregation mismatch: Asset weight identifier missing from risk index: {missing_key}")
            return 1.0  # Default to neutral market risk baseline on structure gaps
        except Exception as error:
            logger.error(f"Vector algebra failure during portfolio beta computation: {error}")
            return 1.0

    @classmethod
    def treynor_ratio(cls, portfolio: Union[Dict[str, Any], pd.Series], asset_risks: pd.DataFrame,
                      rf: float = 0.0) -> float:
        """
        Computes the Treynor Ratio, measuring risk-adjusted return relative to systematic market risk (Beta).
        """
        try:
            portfolio_return = float(portfolio['Returns'])

            # Reconstruct the weights map directly from the series dictionary keys
            # Extracts keys ending in '_weight' while stripping the suffix
            weights_map = {
                key.replace('_weight', ''): float(val)
                for key, val in portfolio.items() if key.endswith('_weight')
            }

            portfolio_beta = cls.calculate_portfolio_beta(weights_map, asset_risks)

            # Enforce a defensive boundary limit to protect against dividing by zero
            if abs(portfolio_beta) < 1e-8:
                logger.warning("Treynor ratio calculation stabilized: Portfolio beta is near zero (Market Neutral).")
                return 0.0

            return (portfolio_return - rf) / portfolio_beta
        except Exception as error:
            logger.error(f"Exception encountered inside Treynor Ratio pipeline: {error}")
            return 0.0

    @staticmethod
    def sharpe_ratio(portfolio: Union[Dict[str, Any], pd.Series], rf: float = 0.0) -> float:
        """
        Computes the corrected Sharpe Ratio, evaluating risk-adjusted returns relative to Total Volatility.
        """
        try:
            portfolio_return = float(portfolio['Returns'])
            portfolio_std = float(portfolio['Volatility'])

            if portfolio_std < 1e-8:
                logger.warning("Sharpe ratio calculation stabilized: Total portfolio volatility collapsed near zero.")
                return 0.0

            return (portfolio_return - rf) / portfolio_std
        except Exception as error:
            logger.error(f"Exception encountered inside Sharpe Ratio pipeline: {error}")
            return 0.0

    @classmethod
    def jensen_alpha(
            cls, portfolio: Union[Dict[str, Any], pd.Series], market_return: float, asset_risks: pd.DataFrame,
            rf: float = 0.0
    ) -> float:
        """
        Computes Jensen's Alpha (Indice), measuring the portfolio's excess abnormal return above the CAPM baseline.
        """
        try:
            portfolio_return = float(portfolio['Returns'])

            weights_map = {
                key.replace('_weight', ''): float(val)
                for key, val in portfolio.items() if key.endswith('_weight')
            }
            portfolio_beta = cls.calculate_portfolio_beta(weights_map, asset_risks)

            # Jensen's Alpha = R_p - [R_f + Beta_p * (R_m - R_f)]
            jensens_alpha = portfolio_return - (rf + portfolio_beta * (market_return - rf))
            return float(jensens_alpha)
        except Exception as error:
            logger.error(f"Exception encountered inside Jensen's Alpha pipeline: {error}")
            return 0.0

    @staticmethod
    def calculate_sml(risk_free_rate: float, market_return: float, beta: float) -> float:
        """
        Computes the expected return step along the Security Market Line (SML) path based on CAPM.
        """
        return risk_free_rate + beta * (market_return - risk_free_rate)
