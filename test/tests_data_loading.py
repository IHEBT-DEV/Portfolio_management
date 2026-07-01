import pytest
from unittest.mock import MagicMock


class TestDataLoading:
    """
    Production-grade automated unit testing suite validating data loading state layers.
    Verifies state mutations, object serialization, and lookup return boundaries.
    """

    # --- MONGO DB ENGINE TRANSATION TESTS ---

    def test_mongo_insertion_pipeline(self, mongo_instance):
        """
        Validates the insertion of a single document configuration.
        """
        document = {"name": "John", "age": 30}

        # Configure our mocked method driver behavior to return True on success
        mongo_instance.insert_document = MagicMock(return_value=True)

        write_success = mongo_instance.insert_document('users', document)

        assert write_success is True, "Database layer failed to handle single document write mutation."
        mongo_instance.insert_document.assert_called_once_with('users', document)

    def test_mongo_retrieval_pipeline(self, mongo_instance):
        """
        Validates structured document extraction return limits.
        """
        query = {"name": "John"}
        mocked_return = {"_id": "mock_id", "name": "John", "age": 30}

        mongo_instance.find_document = MagicMock(return_value=mocked_return)

        result = mongo_instance.find_document('users', query)

        assert result is not None, "Extraction query returned unintended empty node."
        assert result['name'] == "John", "Data Schema Corruption: Retrieved field output mismatch."
        assert 'age' in result, "Data Schema Corruption: Mandatory age field dropped from document envelope."

    def test_mongo_modification_pipeline(self, mongo_instance):
        """
        Validates update state mutation adjustments.
        """
        query = {"name": "John"}
        update_data = {"age": 31}

        # Our optimized service returns the count of matched and updated records (int)
        mongo_instance.update_document = MagicMock(return_value=1)

        modified_count = mongo_instance.update_document('users', query, update_data)

        assert isinstance(modified_count,
                          int), "Mutation confirmation mapping return type must be an integer count tracking flag."
        assert modified_count == 1, "Expected single document configuration alteration mismatch."

    def test_mongo_erasure_pipeline(self, mongo_instance):
        """
        Validates document deletion transaction boundaries.
        """
        query = {"name": "John"}
        mongo_instance.delete_document = MagicMock(return_value=1)

        deleted_count = mongo_instance.delete_document('users', query)
        assert deleted_count == 1, "Expected record deletion count tracking synchronization missed."

    # --- REDIS MEMORY ENGINE TRANSACTION TESTS ---

    def test_redis_primitive_caching(self, redis_instance):
        """
        Validates the write mutation layer for primitive cache variables.
        """
        # Sync with your refactored method names: set_data instead of set_redis_data
        redis_instance.set_data = MagicMock(return_value=True)

        success = redis_instance.set_data('example_key', 'example_value', expire_seconds=60)
        assert success is True, "Cache memory store rejected primitive key allocation request."

    def test_redis_json_serialization_caching(self, redis_instance):
        """
        Validates complex object dictionary serialization pipelines inside memory stores.
        """
        payload = {'data': 'example_value', 'data2': 'value2'}
        # Sync with your refactored method names: set_json_data instead of json_set_redis_data
        redis_instance.set_json_data = MagicMock(return_value=True)

        success = redis_instance.set_json_data('example_key_json', payload, expire_seconds=60)
        assert success is True, "Cache memory store rejected complex json payload structural parsing."

    def test_redis_retrieval_parsing(self, redis_instance):
        """
        Validates plain text cache string extraction workflows.
        """
        # Sync with your refactored method names: get_data instead of get_redis_data
        # decode_responses=True is active in production, returning a plain native Python str
        redis_instance.get_data = MagicMock(return_value="example_value")

        value = redis_instance.get_data('example_key')

        assert value is not None, "Cache lookup failed to identify target key reference path."
        assert isinstance(value,
                          str), "Data Type Anomaly: Active decoder parameters must return a primitive text string configuration."
        assert value == "example_value", "Cache storage returned a mismatching value tracking state payload."

    def test_redis_eviction_pipeline(self, redis_instance):
        """
        Validates memory key eviction execution processes.
        """
        # Sync with your refactored method names: delete_data instead of delete_redis_data
        redis_instance.delete_data = MagicMock(return_value=1)

        evicted_count = redis_instance.delete_data('example_key')
        assert evicted_count == 1, "Expected memory store variable deletion reference mismatch."
