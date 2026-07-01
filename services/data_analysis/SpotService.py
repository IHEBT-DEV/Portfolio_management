import logging
from typing import List, Dict, Any, Optional, Union
from services.data_analysis.helpers.DateHelpers import DateHelpers
from services.data_extraction.ApiExtractor import ApiExtractor
from services.data_extraction.AsyncApiExtractor import AsyncAPIExtractor

logger = logging.getLogger(__name__)


class SpotServices:
    """
    Orchestration core for market historical data extraction pipelines.
    Dynamically switches routing pathways between synchronous and high-concurrency asynchronous execution models.
    """

    # Primary immutable endpoint configuration schemas
    BINANCE_BASE_URL = "https://api.binance.com"
    BINANCE_KLINES_ENDPOINT = "api/v3/klines"

    def __init__(self, api_extractor: Optional[ApiExtractor] = None,
                 async_extractor: Optional[AsyncAPIExtractor] = None):
        """
        Initializes the Core Analysis Orchestrator.
        Leverages dependency injection singletons to reuse open TCP socket connections.
        """
        self.api = api_extractor or ApiExtractor()
        self.async_api = async_extractor or AsyncAPIExtractor()

    def get_spot_api(
            self,
            market: str = 'BINANCE',
            currency: str = 'BTCUSDT',
            interval: str = '1D',
            start_date: Optional[Union[int, str]] = None,
            end_date: Optional[Union[int, str]] = None
    ) -> Optional[Any]:
        """
        Public orchestrator entrypoint to execute historical pricing queries across different exchanges.
        """
        market_key = market.strip().upper()

        if market_key == 'BINANCE':
            return self._get_spot_binance_api(currency.upper(), interval, start_date, end_date)

        logger.warning(f"Data analysis route aborted: Unrecognized market target specified [{market}].")
        return None

    def _get_spot_binance_api(
            self,
            currency: str,
            interval: str,
            start_date: Optional[Union[int, str]] = None,
            end_date: Optional[Union[int, str]] = None
    ) -> Optional[Any]:
        """
        Internal retrieval logic targeting the Binance API architecture.
        """
        # Defensive interval formatting layer: Binance expects days ('1D') and months ('1M') in UPPERCASE,
        # but minutes/hours in lowercase. If it ends with 'd', 'w', or 'm' (month token check), capitalize it.
        clean_interval = interval
        if interval[-1].lower() in ['d', 'w', 'm']:
            clean_interval = f"{interval[:-1]}{interval[-1].upper()}"
        else:
            clean_interval = interval.lower()

        target_url = f"{self.BINANCE_BASE_URL}/{self.BINANCE_KLINES_ENDPOINT}"

        # PATHWAY A: Multi-part historical data chunk processing loop (High-Concurrency Async)
        if start_date and end_date:
            try:
                list_dates = DateHelpers.list_start_end_chunks(start_date=start_date, end_date=end_date, interval=interval)
                if not list_dates:
                    logger.warning(f"Date chunk configuration empty for ranges: {start_date} to {end_date}.")
                    return []

                urls_batch = []
                for chunk in list_dates:
                    spot_params = {
                        "symbol": currency,
                        "interval": clean_interval,
                        "startTime": chunk['start_date'],
                        "limit": chunk['limit']
                    }
                    urls_batch.append({'url': target_url, 'params': spot_params})

                # Execute optimized bulk extraction
                return self.async_api.get_data(urls_batch)

            except Exception as async_pipeline_error:
                logger.error(
                    f"Asynchronous pipeline extraction anomaly hit inside SpotServices: {async_pipeline_error}")
                return []

        # PATHWAY B: Real-time immediate evaluation query (Synchronous)
        else:
            spot_params = {
                "symbol": currency,
                "interval": clean_interval,
                "limit": 1000
            }
            try:
                return self.api.get_data(self.BINANCE_BASE_URL, self.BINANCE_KLINES_ENDPOINT, spot_params)
            except Exception as sync_pipeline_error:
                logger.error(f"Synchronous pipeline extraction anomaly hit inside SpotServices: {sync_pipeline_error}")
                return None
