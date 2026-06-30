import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)


class AssetRiskCalculation:
    """
    Advanced asset risk decomposition engine.
    Applies the Single-Index Model (SIM) to separate systematic market risk from specific asset risk.
    """

    @staticmethod
    def calculate_asset_risk_profile(
            market_index_df: pd.DataFrame, close_df: pd.DataFrame, asset_symbol: str
    ) -> Dict[str, Any]:
        """
        Decomposes total asset risk into its systematic and specific components using OLS regression.

        :param market_index_df: DataFrame containing the 'market_index' column baseline.
        :param close_df: DataFrame containing historical closing prices for assets.
        :param asset_symbol: Target column asset label to evaluate (e.g., 'BTCUSDT').
        :return: A standardized dictionary profile containing risk analytics keys.
        """
        if asset_symbol not in close_df.columns or 'market_index' not in market_index_df.columns:
            logger.error(f"Risk calculation aborted: Mandatory tracking keys missing for asset [{asset_symbol}].")
            return {}

        try:
            # 1. Isolate percentage returns using forward-filling to maintain timeline alignment
            asset_returns = close_df[asset_symbol].pct_change().ffill()
            market_returns = market_index_df['market_index'].pct_change().ffill()

            # 2. Build an isolated local tracking frame to bypass special character formula crashes
            regression_df = pd.DataFrame({
                'asset_target': asset_returns,
                'market_baseline': market_returns
            }).dropna()

            if len(regression_df) < 5:
                logger.warning(f"Insufficient aligned timeline steps to evaluate risk profile for [{asset_symbol}].")
                return {}

            # 3. Execute Simple Regression model: asset_target = alpha + beta * market_baseline + error
            model = smf.ols(formula="asset_target ~ market_baseline", data=regression_df).fit()

            # 4. Extract raw model parameters safely
            beta_asset = float(model.params.get('market_baseline', 0.0))
            residual_variance = float(model.resid.var(ddof=1))  # Sample variance bounds
            market_variance = float(regression_df['market_baseline'].var(ddof=1))

            # 5. Execute Single-Index Model risk decomposition calculations
            # Systematic Risk = sqrt(Var(Market) * Beta^2)
            systematic_risk = np.sqrt(max(market_variance * (beta_asset ** 2), 0.0))

            # Specific Risk = Residual Standard Deviation
            specific_risk = np.sqrt(max(residual_variance, 0.0))

            # Total Risk = sqrt(Specific_Risk^2 + Systematic_Risk^2)
            total_risk = np.sqrt((specific_risk ** 2) + (systematic_risk ** 2))

            # 6. Format metrics as clean percentages matching your display parameters
            return {
                'Asset': asset_symbol.upper(),
                'Beta_asset': float(round(beta_asset, 4)),
                'Class': 'Aggressive' if beta_asset > 1.0 else 'Defensive',
                'Total_risk': float(round(total_risk * 100.0, 4)),
                'Systematic_risk': float(round(systematic_risk * 100.0, 4)),
                'Specific_risk': float(round(specific_risk * 100.0, 4))
            }

        except Exception as computation_fault:
            logger.error(
                f"Mathematical execution exception inside asset risk analyzer for [{asset_symbol}]: {computation_fault}")
            return {}
