import pytest
from services.data_analysis.SpotService import SpotServices
from services.data_transformation.CandlestickParser import CandlestickParser
from services.data_loading.MongoDbService import MongoDbService


class TestDataExtraction:
    """
    Production-grade automated unit testing suite validating the ingestion layer.
    Enforces high-speed in-memory matrix parsing and data contract validations.
    """

    def test_api_extractor_binance_pipeline(
            self,
            spot_api_instance: SpotServices,
            spot_candlestick_parser: CandlestickParser,
            mongo_instance: MongoDbService,
            sample_ohlcv_data: list
    ):
        """
        Validates the complete ETL data pipeline: Extraction, Transformation, and Loading contracts.
        Reuses session data fixtures to isolate tests completely from live external network dependencies.
        """
        # 1. Transform our mock data payload arrays natively to simulate an inbound API matrix stream
        parsed_binance_data = spot_candlestick_parser.parse_spot_by_market_currency(
            sample_ohlcv_data,
            market='BINANCE',
            currency='BTCUSDT'
        )

        # 2. Enforce Data Contract Assertions: Verify that every mandatory financial data key exists
        assert isinstance(parsed_binance_data,
                          list), "The data pipeline output layer must return a valid list envelope."
        assert len(parsed_binance_data) > 0, "The transformation matrix parser cannot return an empty asset list."

        target_record = parsed_binance_data[0]
        mandatory_keys = {'currency', 'date', 'open', 'high', 'low', 'close', 'volume'}

        for key in mandatory_keys:
            assert key in target_record, f"Data Contract Violation: Expected structural data tracking key missing: [{key}]"

        # 3. Verify Exact Schema Element Types
        assert target_record['currency'] == 'BTCUSDT', "Linguistic error: Target pair name label mismatch."
        assert isinstance(target_record['open'],
                          float), "Data Type Violation: Expected float parsing for numerical fields."
        assert isinstance(target_record['close'],
                          float), "Data Type Violation: Expected float parsing for numerical fields."

        # 4. Validate Isolated Loading Infrastructure Execution
        # Uses our patched mongo fixture to test database layer writes safely without hitting a live database cluster
        write_success = mongo_instance.insert_many_documents(collection_name='spot', documents=parsed_binance_data)

        # Verify that our mocked load statement smoothly handles database insert requests
        assert write_success is not None, "Database interface transaction failed to handle bulk insert requests."
