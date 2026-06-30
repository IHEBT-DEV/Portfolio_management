import asyncio
import logging
from typing import List, Dict, Any, Union
import aiohttp

# Configure standard structured production logging
logger = logging.getLogger(__name__)


class AsyncAPIExtractor:
    def __init__(self, max_concurrent_requests: int = 10, timeout_seconds: int = 30):
        """
        High-throughput asynchronous API Extractor engine.

        :param max_concurrent_requests: Firm limit on simultaneous outbound connections.
        :param timeout_seconds: Hard cutoff limit to prevent network socket hang-ups.
        """
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def get_data(self, endpoint_configs: List[Dict[str, Any]]) -> List[Union[Dict[str, Any], Exception]]:
        """
        Public entrypoint to batch fetch multiple endpoint arrays asynchronously.
        """
        if not endpoint_configs:
            return []

        try:
            return await self._global_fetch_urls(endpoint_configs)
        except Exception as error:
            logger.error(f"Fatal operational extraction failure: {error}", exc_info=True)
            raise error

    async def _global_fetch_urls(self, endpoint_configs: List[Dict[str, Any]]) -> List[
        Union[Dict[str, Any], Exception]]:
        # Single persistent session architecture to maximize TCP connection reuse
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            tasks = []
            for config in endpoint_configs:
                url = config.get("url")
                params = config.get("params", {})

                if not url:
                    logger.warning("Skipping extraction entry: Target URL missing parameter mapping.")
                    continue

                # Schedule task bounded tightly within our semaphore throttle
                task = asyncio.ensure_future(self._fetch_url_with_throttle(session, url, params))
                tasks.append(task)

            # return_exceptions=True prevents one failed API point from destroying the entire batch
            return await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_url_with_throttle(self, session: aiohttp.ClientSession, url: str, params: Dict[str, Any]) -> Dict[
        str, Any]:
        # Context block enforces concurrency throttling limit rules
        async with self.semaphore:
            try:
                async with session.get(url, params=params) as response:
                    # Explicit HTTP status error propagation
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ClientResponseError as http_error:
                logger.error(f"HTTP Status Error encountered for URL [{url}]: Status {http_error.status}")
                raise http_error
            except asyncio.TimeoutError as timeout_error:
                logger.error(f"Network Timeout limit breached for request: [{url}]")
                raise timeout_error
            except Exception as execution_error:
                logger.error(f"Unexpected connection exception at endpoint [{url}]: {execution_error}")
                raise execution_error
