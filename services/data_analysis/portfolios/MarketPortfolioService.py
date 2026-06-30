import logging
from typing import Dict, List, Any
import pandas as pd

logger = logging.getLogger(__name__)


class MarketPortfolioService:
    """
    Enterprise-grade Market Portfolio Analysis Engine.
    Orchestrates index indexing adjustments and processes weighted market index benchmarks.
    """

    def __init__(self, market_df: pd.DataFrame):
        """
        Initializes the Market Portfolio Service tracking layers.

        :param market_df: Master DataFrame containing chronological asset price timelines.
        """
        # Maintain a clean, immutable master reference to protect data state integrity
        self._master_df = market_df.copy()

    @property
    def current_market_state(self) -> pd.DataFrame:
        """
        Returns a fresh copy of the master data frame context to prevent cross-module pollution.
        """
        return self._master_df.copy()

    def get_last_prices(self) -> List[Any]:
        """
        Extracts the most recent asset price vector from the chronological timeline matrix.
        """
        if self._master_df.empty:
            logger.warning("Pricing vector lookup failed: Selected source market data frame is empty.")
            return []
        return self._master_df.iloc[-1].tolist()

    @staticmethod
    def market_index_df(market_df_weighted: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates a weighted asset matrix down into a single market index benchmark tracking frame.
        """
        if market_df_weighted.empty:
            return pd.DataFrame(columns=['market_index'])

        try:
            # Create isolated local frame context to prevent input array mutations
            index_df = pd.DataFrame(index=market_df_weighted.index)
            # Row-by-row matrix summation aggregation pass
            index_df['market_index'] = market_df_weighted.sum(axis=1)
            return index_df
        except Exception as matrix_fault:
            logger.error(f"Matrix algebraic summation anomaly inside market_index_df: {matrix_fault}")
            return pd.DataFrame(index=market_df_weighted.index, columns=['market_index']).fillna(0.0)

    def weighted_df(self, index_weights: Dict[str, float]) -> pd.DataFrame:
        """
        Applies a weight matrix transformation over asset paths.
        Returns a fresh DataFrame containing only weighted asset price timelines.
        """
        if self._master_df.empty or not index_weights:
            return pd.DataFrame()

        try:
            # Generate a fresh local working frame context safely
            weighted_output_df = pd.DataFrame(index=self._master_df.index)

            for key, weight_value in index_weights.items():
                if key not in self._master_df.columns:
                    logger.warning(
                        f"Transformation gap: Column selector [{key}] missing from asset timeline. Skipping allocation.")
                    continue

                # Apply allocation scalar math vector transformations directly
                weighted_output_df[f"{key}_WEIGHTED"] = self._master_df[key] * weight_value

            return weighted_output_df

        except Exception as calculation_fault:
            logger.error(f"Mathematical execution exception inside weighted_df computation layer: {calculation_fault}")
            return pd.DataFrame()
