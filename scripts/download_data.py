import json
import logging
import sys
import urllib.request
from pathlib import Path

# Add src to python path for internal imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clickbait_spoiling.logging_config import setup_logging
from clickbait_spoiling.utils.io_utils import ensure_dir

logger = logging.getLogger(__name__)

ZENODO_API_URL = "https://zenodo.org/api/records/6362726"


def download_file(url: str, dest_path: Path):
    """Download file with a simple text progress indicator."""
    logger.info(f"Downloading {url} -> {dest_path}")
    urllib.request.urlretrieve(url, dest_path)


def verify_line_count(file_path: Path, expected: int) -> bool:
    """Verify that file has exactly the expected number of JSON Lines."""
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count == expected


def main():
    setup_logging("outputs/logs", "download_data")
    raw_dir = ensure_dir("data/raw")

    logger.info("Querying Zenodo API for Clickbait Spoiling Corpus...")
    try:
        req = urllib.request.Request(
            ZENODO_API_URL, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req) as response:
            metadata = json.loads(response.read().decode())

        files_info = metadata.get("files", [])
        if not files_info:
            logger.error("No files found in Zenodo metadata response.")
            sys.exit(1)

        downloads = {"training.jsonl": 3200, "validation.jsonl": 800}

        for file_info in files_info:
            filename = file_info["key"]
            if filename in downloads:
                download_url = file_info["links"]["self"]
                dest_path = raw_dir / filename

                # Download file
                download_file(download_url, dest_path)

                # Verify downloaded file Integrity
                expected_lines = downloads[filename]
                if verify_line_count(dest_path, expected_lines):
                    logger.info(
                        f"Successfully verified integrity of {filename} ({expected_lines} records)."
                    )
                else:
                    logger.error(
                        f"Integrity check failed for {filename}. Line counts mismatch. Re-run download."
                    )
                    sys.exit(1)

        logger.info("Dataset retrieval pipeline finished successfully.")

    except Exception as e:
        logger.error(f"An error occurred during dataset download: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
