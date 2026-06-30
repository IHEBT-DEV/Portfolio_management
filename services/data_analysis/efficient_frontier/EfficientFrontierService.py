import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class EfficientFrontierService:
    """
    High-performance quantitative portfolio optimization engine.
    Computes optimal asset weighting allocations along the Markowitz Efficient Frontier.
    """

    def __init__(self, returns: List[float], cov_matrix: List[List[float]], periods_per_year: int = 365):
        """
        Initializes the Efficient Frontier service tracking parameters.

        :param returns: Expected returns matrix vector for each asset slot.
        :param cov_matrix: Computed cross-asset covariance matrix arrays.
        :param periods_per_year: Annualization conversion factor (Crypto default: 365).
        """
        self.returns = np.array(returns, dtype=float)
        self.cov_matrix = np.array(cov_matrix, dtype=float)
        self.annualizing_factor = periods_per_year

    def objective_function(self, weights: np.ndarray) -> float:
        """
        Core objective helper function to optimize allocations by maximizing the Sharpe Ratio.
        Returns the negative Sharpe Ratio for minimization loops.
        """
        portfolio_return = np.sum(self.returns * weights)

        # Calculate annualized volatility using matrix dot product multiplications
        variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        # Ensure variance stays mathematically non-negative before applying square root
        portfolio_volatility = np.sqrt(max(variance, 0.0)) * np.sqrt(self.annualizing_factor)

        # Prevent division-by-zero errors via low-level numerical stabilization boundaries
        if portfolio_volatility < 1e-8:
            return 0.0

        sharpe_ratio = portfolio_return / portfolio_volatility
        return -sharpe_ratio

    def efficient_frontier_portfolios(self, grid_resolution: int = 50) -> List[List[float]]:
        """
        Constructs the efficient frontier allocations matrix by matching incremental target returns.
        """
        num_assets = len(self.returns)
        if num_assets == 0:
            return []

        # Initial uniform weight guess allocation vector (Equal Weights)
        initial_weights = np.ones(num_assets) / num_assets

        # Enforce boundary layout rules: Long-only spot mechanics (Weights must live between 0.0 and 1.0)
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))

        # Generate target returns steps along the distribution boundary limits
        target_returns = np.linspace(min(self.returns), max(self.returns), num=grid_resolution)
        portfolios: List[List[float]] = []

        for target_return in target_returns:
            # Enforce clean system constraints:
            # 1. Sum of all weights must equal 1.0
            # 2. Portfolio return must hit the incremental target point exactly
            constraints = (
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w: np.sum(self.returns * w) - target_return}
            )

            try:
                result = minimize(
                    self.objective_function,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 100}
                )

                if result.success:
                    # Clean up low-level optimization floating variations (round tiny negatives to zero)
                    sanitized_weights = np.where(result.x < 1e-5, 0.0, result.x)
                    # Normalize weights to ensure absolute structural sum parity
                    if np.sum(sanitized_weights) > 0:
                        sanitized_weights = sanitized_weights / np.sum(sanitized_weights)
                    portfolios.append(sanitized_weights.tolist())
                else:
                    # Fallback to initial guess weights if optimizer fails to reach absolute convergence
                    portfolios.append(initial_weights.tolist())

            except Exception as optimization_fault:
                logger.error(
                    f"Mathematical execution optimization exception at target parameter [{target_return}]: {optimization_fault}")
                portfolios.append(initial_weights.tolist())

        return portfolios

    def portfolio_returns(self, portfolios: List[List[float]]) -> List[float]:
        """
        Computes mathematical expected returns vector mappings for an array matrix of portfolio weights.
        """
        return [float(np.sum(self.returns * np.array(w))) for w in portfolios]

    def portfolio_volatilities(self, portfolios: List[List[float]]) -> List[float]:
        """
        Computes historical annualized risks vector mappings for an array matrix of portfolio weights.
        """
        volatilities: List[float] = []
        for w_list in portfolios:
            w = np.array(w_list)
            variance = np.dot(w.T, np.dot(self.cov_matrix, w))
            vol = np.sqrt(max(variance, 0.0)) * np.sqrt(self.annualizing_factor)
            volatilities.append(float(vol))
        return volatilities
