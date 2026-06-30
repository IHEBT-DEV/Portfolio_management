import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from services.data_extraction.ApiExtractor import ApiExtractor

logger = logging.getLogger(__name__)


class MarketCapService:
    """
    Enterprise-grade market capitalization and Index Allocation analysis engine.
    Orchestrates macro coin metrics and implements Capital Market Line calculations.
    """

    COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, currencies: List[str], api_extractor: Optional[ApiExtractor] = None):
        """
        Initializes the Market Cap analyzer.

        :param currencies: List of core asset symbols to track (e.g., ['BTC', 'ETH']).
        :param api_extractor: Injectable synchronous connection broker singleton.
        """
        self.currencies = [c.strip().upper() for c in currencies]
        self.api = api_extractor or ApiExtractor()

    @staticmethod
    def currency_name_by_symbol(currency_symbol: str) -> Optional[str]:
        """
        Maps standard shorthand market tickers securely to CoinGecko's full text slugs.
        """
        clean_symbol = currency_symbol.strip().lower()

        # Normalize pairs out to base assets
        if clean_symbol in ['btc', 'btcusdt']:
            return 'bitcoin'
        if clean_symbol in ['eth', 'ethusdt']:
            return 'ethereum'

        logger.warning(f"Linguistic lookup missed: Asset identifier mapping key not registered [{currency_symbol}].")
        return None

    @staticmethod
    def market_cap_by_currency(all_outstanding_shares: Dict[str, float], prices: Dict[str, float]) -> Dict[str, float]:
        """
        Computes the market capitalization for each asset based on current prices.
        """
        market_cap: Dict[str, float] = {}
        for key, shares in all_outstanding_shares.items():
            current_price = prices.get(key)
            if current_price is None:
                logger.warning(
                    f"Calculations variance gap: Price vector absent for ticker token [{key}]. Skipping cap computation.")
                continue
            market_cap[key] = shares * current_price
        return market_cap

    @staticmethod
    def total_market_cap(market_cap: Dict[str, float]) -> float:
        """
        Computes the aggregated sum total valuation across a market capitalization mapping dictionary.
        """
        return float(sum(market_cap.values()))

    @staticmethod
    def calculate_cml(
            rf: float, expected_market_return: float, market_volatility: float, grid_points: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Capital Market Line (CML) vector arrays.
        Enforces a defensive denominator stabilization layer to guarantee division safety.
        """
        # Ensure return difference spans a valid mathematical division range
        denominator = expected_market_return - rf
        if abs(denominator) < 1e-8 or market_volatility < 1e-8:
            logger.error(
                "Mathematical anomaly inside CML pipeline: Return variance or volatility limits collapsed near zero.")
            empty_vector = np.array([])
            return empty_vector, empty_vector

        cml_returns = np.linspace(rf, expected_market_return, num=grid_points)
        # Vectorized linear transformation computation formula loop
        cml_volatility = ((cml_returns - rf) / denominator) * market_volatility

        return cml_returns, cml_volatility

    def outstanding_shares_by_currency(self) -> Dict[str, float]:
        """
        Fetches live circulating supply metrics from the external CoinGecko data core.
        Returns a clean tracking map of normalized pair labels on success.
        """
        outstanding_shares_all: Dict[str, float] = {}

        for currency in self.currencies:
            # Drop the trading pair suffix formatting to track raw asset identities cleanly
            base_symbol = currency.replace("USDT", "").lower()
            currency_name = self.currency_name_by_symbol(base_symbol)

            if not currency_name:
                continue

            try:
                endpoint = f"coins/{currency_name}"
                # Request pricing tracking packets safely
                raw_payload = self.api.get_data(self.COINGECKO_BASE_URL, endpoint)

                if not raw_payload or not isinstance(raw_payload, dict):
                    logger.error(
                        f"Inbound ingestion error: Failed retrieving valid tracking data for [{currency_name}]")
                    continue

                # Defensive fallback layout navigation to avoid unhandled KeyErrors
                market_data = raw_payload.get('market_data', {})
                market_cap_dict = market_data.get('market_cap', {})
                circulating_supply = market_cap_dict.get(base_symbol)

                if circulating_supply is None:
                    logger.warning(
                        f"Metric logging gap: Circulating data token missing for token asset [{base_symbol}] inside JSON payload.")
                    continue

                target_key = f"{base_symbol.upper()}USDT"
                outstanding_shares_all[target_key] = float(circulating_supply)

            except Exception as execution_fault:
                logger.error(
                    f"Unexpected connectivity error processing asset vector metrics for [{currency}]: {execution_fault}")
                continue

        return outstanding_shares_all

    def market_index_weights(self, market_cap: Dict[str, float]) -> Dict[str, float]:
        """
        Computes capitalization-weighted asset index allocations.
        """
        total = self.total_market_cap(market_cap)
        if total <= 0:
            logger.error(
                "Weights calculation aborted: Total index market capitalization evaluates to non-positive bounds.")
            return {key: 0.0 for key in market_cap.keys()}

        return {f"{key}": value / total for key, value in market_cap.items()}
