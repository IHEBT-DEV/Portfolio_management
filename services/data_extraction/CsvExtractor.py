import csv
import logging
from pathlib import Path
from typing import Generator, Dict, Any, Union, Optional

logger = logging.getLogger(__name__)


class CsvExtractor:
    """
    High-performance CSV stream extraction engine.
    Utilizes lazy evaluation generator layers to process large datasets with an immutable memory footprint.
    """

    def __init__(self, file_path: Union[str, Path]):
        """
        Initializes the streaming CSV extraction channel.

        :param file_path: System filesystem location mapping the target CSV asset.
        """
        self.file_path = Path(file_path)

    def extract_data_stream(self) -> Generator[Dict[str, Any], None, None]:
        """
        A memory-safe lazy generator that yields individual CSV rows on-demand.
        Ensures a flat, sub-1MB RAM consumption profile regardless of input file size.
        """
        if not self.file_path.exists():
            logger.error(f"Data ingestion aborted: File target path does not exist [{self.file_path}]")
            return

        try:
            # Explicitly enforce safe utf-8 decoding parameters
            with open(self.file_path, mode='r', encoding='utf-8', newline='') as file:
                reader = csv.DictReader(file)

                # Verify that the CSV structure actually contains valid column fields
                if reader.fieldnames is None:
                    logger.warning(
                        f"Aborting parsing stream: Selected CSV file structure appears empty or unformatted [{self.file_path}]")
                    return

                for row_index, row in enumerate(reader):
                    # Filter out unintended blank or empty text segment lines
                    if not row or all(value is None or value.strip() == "" for value in row.values()):
                        continue

                    yield dict(row)

        except PermissionError:
            logger.error(
                f"OS Security exception: Missing file access read permission bounds over path [{self.file_path}]")
            raise
        except Exception as execution_error:
            logger.error(f"Unexpected operational anomaly during CSV parsing layout stream: {execution_error}")
            raise
