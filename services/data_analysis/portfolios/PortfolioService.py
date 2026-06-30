import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioService:
    """
    High-performance Portfolio Risk and Simulation Engine.
    Executes vectorized Monte Carlo allocation asset testing routines and computes annualized metrics.
    """

    def __init__(self, close_df: pd.DataFrame, periods_per_year: int = 365):
        """
        Initializes the Portfolio Service.

        :param close_df: Chronological DataFrame containing raw closing asset price sequences.
        :param periods_per_year: Annualization factor (Default: 365 matching crypto daily profiles).
        """
        self._master_df = close_df.copy()
        self.annualizing_factor = periods_per_year

    @staticmethod
    def yearly_nb_periods(granularity_seconds: int = 86400, trading_days: int = 365) -> float:
        """
        Computes the theoretical number of trading period steps contained within a standard market year context.
        """
        day_seconds = 86400
        # Prevent division-by-zero on corrupted inputs via safety floors
        safe_granularity = max(granularity_seconds, 1)
        daily_periods = day_seconds / safe_granularity
        return float(trading_days * daily_periods)

    @staticmethod
    def asset_returns_volatility(expected_returns: pd.Series, annualized_volatility: pd.Series) -> pd.DataFrame:
        """
        Combines isolated expected asset return and volatility maps into a clean, uniform overview frame.
        """
        assets = pd.concat([expected_returns, annualized_volatility], axis=1)
        assets.columns = ['Returns', 'Volatility']
        return assets

    def _get_clean_returns(self) -> pd.DataFrame:
        """
        Internal data transformation helper. Computes standardized percentage returns vectors
        using defensive forward-filling to maintain clean data alignment.
        """
        # Execute percentage variance calculation and isolate the initial NaN gap safely via forward filling
        return self._master_df.pct_change().ffill().dropna()

    def expected_return_annualized(self) -> pd.Series:
        """
        Computes the mathematically compounded annualized expected return across all tracked assets.
        """
        try:
            returns_df = self._get_clean_returns()
            if returns_df.empty:
                return pd.Series(dtype=float)

            # Compound return formula matrix application: (1 + mean_return) ** factor - 1
            mean_returns = returns_df.mean()
            return mean_returns.apply(lambda x: float((1.0 + x) ** self.annualizing_factor - 1.0))
        except Exception as error:
            logger.error(f"Mathematical execution exception inside annualized return compiler: {error}")
            return pd.Series(dtype=float)

    def volatility_annualized(self) -> pd.Series:
        """
        Computes the annualized statistical volatility standard deviation metric for each asset vector.
        """
        try:
            returns_df = self._get_clean_returns()
            if returns_df.empty:
                return pd.Series(dtype=float)

            std_returns = returns_df.std()
            return std_returns.apply(lambda x: float(x * np.sqrt(self.annualizing_factor)))
        except Exception as error:
            logger.error(f"Mathematical statistical exception inside volatility annualization layer: {error}")
            return pd.Series(dtype=float)

    def covariance_matrix(self) -> pd.DataFrame:
        """
        Generates the synchronized cross-asset relational covariance matrix tracking frame.
        """
        try:
            returns_df = self._get_clean_returns()
            return returns_df.cov()
        except Exception as error:
            logger.error(f"Matrix algebraic covariance transformation anomaly: {error}")
            return pd.DataFrame()

    def portfolios_simulation(
            self, expected_returns: pd.Series, cov_matrix: pd.DataFrame, num_portfolios: int = 1000
    ) -> pd.DataFrame:
        """
        Executes a vectorized Monte Carlo simulation to test randomized asset weight configurations.
        Utilizes NumPy matrix dot products to achieve high computational speed.
        """
        if expected_returns.empty or cov_matrix.empty or num_portfolios <= 0:
            return pd.DataFrame()

        try:
            num_assets = len(self._master_df.columns)
            columns_list = self._master_df.columns.tolist()

            # Convert inputs directly to low-level NumPy arrays to bypass pandas indexing overhead
            exp_ret_arr = expected_returns.values
            cov_mat_arr = cov_matrix.values

            # Vectorized Matrix Generation: Compute all random allocations in one single step
            raw_weights = np.random.random((num_portfolios, num_assets))
            # Normalize row sums to exactly 1.0 to enforce long-only spot allocation constraints
            weights_matrix = raw_weights / np.sum(raw_weights, axis=1, keepdims=True)

            # High-speed array dot product calculation loops
            portfolio_returns = np.dot(weights_matrix, exp_ret_arr)

            # Compute tracking variance profiles natively via linear algebra expressions
            portfolio_volatilities = np.zeros(num_portfolios)
            sqrt_factor = np.sqrt(self.annualizing_factor)

            for i in range(num_portfolios):
                w = weights_matrix[i]
                variance = np.dot(w.T, np.dot(cov_mat_arr, w))
                # Protective boundary adjustment to shield calculations from imaginary number faults
                portfolio_volatilities[i] = np.sqrt(max(variance, 0.0)) * sqrt_factor

            # Standardize and populate the structural output tracking layout map
            data_payload: Dict[str, Any] = {
                'Returns': portfolio_returns,
                'Volatility': portfolio_volatilities
            }

            # Populate separate allocation weight track columns seamlessly for each asset token
            for asset_idx, symbol in enumerate(columns_list):
                data_payload[f"{symbol}_weight"] = weights_matrix[:, asset_idx]

            # Generate final target output frame and arrange by returns performance metrics profiles
            simulation_df = pd.DataFrame(data_payload).sort_values('Returns', ascending=False)
            return simulation_df

        except Exception as simulation_fault:
            logger.error(f"High-frequency Monte Carlo simulation loop exception captured: {simulation_fault}")
            return pd.DataFrame()
