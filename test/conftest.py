import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from services.data_loading.MongoDbService import MongoDbService
from services.data_loading.RedisService import RedisService
from services.data_transformation.CandlestickParser import CandlestickParser
from services.data_analysis.SpotService import SpotServices
from services.data_analysis.technical_analysis.Metrics import Metrics
from services.data_analysis.technical_analysis.TradingSignals import TradingSignals


@pytest.fixture(scope='session')
def sample_ohlcv_data() -> list:
    """
    Shared session fixture providing a standardized, mock Binance-style OHLCV raw data array.
    Structure: [Timestamp, Open, High, Low, Close, Volume]
    """
    return [
        [1672531200000, "16500.0", "16600.0", "16400.0", "16550.0", "100.0"],
        [1672617600000, "16550.0", "16800.0", "16500.0", "16700.0", "150.0"],
        [1672704000000, "16700.0", "16750.0", "16200.0", "16300.0", "200.0"],
        [1672790400000, "16300.0", "16450.0", "16100.0", "16400.0", "120.0"],
        [1672876800000, "16400.0", "16600.0", "16350.0", "16500.0", "80.0"]
    ]


@pytest.fixture(scope='session')
def sample_dataframe() -> pd.DataFrame:
    """
    Shared session fixture returning a standardized, pandas DataFrame layout for analytical tools.
    """
    return pd.DataFrame({
        'date': pd.date_range(start="2026-01-01", periods=5, freq='D'),
        'open': [100.0, 102.0, 101.0, 105.0, 104.0],
        'high': [103.0, 106.0, 102.0, 107.0, 106.0],
        'low': [99.0, 101.0, 98.0, 103.0, 102.0],
        'close': [102.0, 101.0, 105.0, 104.0, 106.0],
        'volume': [1000.0, 1500.0, 1200.0, 1800.0, 1400.0],
        'spread': [3.0, 5.0, 4.0, 4.0, 4.0]
    })


@pytest.fixture()
def mongo_instance():
    """
    Returns a securely isolated MongoDbService instance protected by low-level client patching.
    Insulates testing environments from real infrastructure socket connection requirements.
    """
    with patch('pymongo.MongoClient') as mock_client:
        # Prevent the interior initialization ping check from throwing connectivity faults
        mock_client.return_value.server_info.return_value = {"ok": 1.0}
        instance = MongoDbService(connection_uri="mongodb://mock:27017", database_name="test_db")
        yield instance


@pytest.fixture()
def redis_instance():
    """
    Returns an isolated, memory-safe RedisService mock context.
    """
    with patch('redis.Redis') as mock_redis:
        # Satisfy the initialization structural ping validation check instantly
        mock_redis.return_value.ping.return_value = True
        instance = RedisService(host="mock_cache", port=6379, db=0)
        yield instance


@pytest.fixture(scope='session')
def spot_api_instance():
    """
    Returns a unified, session-scoped SpotServices orchestrator interface.
    """
    return SpotServices()


@pytest.fixture(scope='session')
def spot_candlestick_parser():
    """
    Returns a unified, session-scoped CandlestickParser utility layer.
    """
    return CandlestickParser()
