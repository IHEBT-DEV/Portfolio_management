import logging
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)


class ApiExtractor:
    """
    Synchronous HTTP data extraction engine.
    Optimizes network connection reuse via persistent state sessions with bounded timeouts.
    """

    def __init__(self, timeout_seconds: int = 15):
        """
        Initializes the synchronous extraction broker instance.

        :param timeout_seconds: Maximum network constraint window before connection cutoff.
        """
        self.timeout = timeout_seconds
        # Establish persistent TCP connection reuse pool
        self.session = requests.Session()

    def get_data(
            self,
            base_url: str,
            endpoint: str,
            params: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Executes a synchronous GET query to standard structural API routes.
        Returns parsed JSON payloads on success, or None if network execution fails.
        """
        # Formulate and format targeted clean request paths
        clean_base = base_url.rstrip('/')
        clean_endpoint = endpoint.lstrip('/')
        target_url = f"{clean_base}/{clean_endpoint}"

        try:
            response = self.session.get(
                target_url,
                headers=headers,
                params=params,
                timeout=self.timeout
            )
            # Instantly catch and bubble up all non-2xx HTTP anomalies
            response.raise_for_status()

            return response.json()

        except requests.exceptions.HTTPError as http_error:
            logger.error(
                f"HTTP Connection Exception encountered on route [{target_url}]: Status {response.status_code}")
            return None
        except requests.exceptions.Timeout as timeout_error:
            logger.error(f"Network Timeout parameter limit exceeded for request route: [{target_url}]")
            return None
        except requests.exceptions.RequestException as general_network_error:
            logger.error(
                f"Structural HTTP protocol communication anomaly on route [{target_url}]: {general_network_error}")
            return None
        except ValueError as json_serialization_error:
            logger.error(f"Payload Transformation Anomaly: Failed parsing JSON sequence data from [{target_url}]")
            return None

    def close(self) -> None:
        """
        Cleanly closes and releases all downstream pool sockets in the open connection session.
        """
        self.session.close()
