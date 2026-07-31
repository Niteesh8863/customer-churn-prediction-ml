"""Dataset acquisition and loading for the IBM Telco Customer Churn dataset."""

from __future__ import annotations

import pandas as pd
import requests

from src.config import DATASET_URL, RAW_DATA_FILE
from src.utils import get_logger

logger = get_logger(__name__)


def download_dataset(url: str = DATASET_URL, destination=RAW_DATA_FILE) -> None:
    """Download the Telco Customer Churn CSV if it is not already present.

    Parameters
    ----------
    url:
        Source URL for the raw CSV file.
    destination:
        Local path to save the file to.

    Raises
    ------
    RuntimeError
        If the download fails and no local copy already exists.
    """
    if destination.exists():
        logger.info("Dataset already present at %s, skipping download.", destination)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading dataset from %s ...", url)
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Automatic download failed ({exc}). Please manually download the "
            "IBM Telco Customer Churn dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv) "
            f"and place it at: {destination}"
        ) from exc

    destination.write_bytes(response.content)
    logger.info("Dataset saved to %s (%d bytes).", destination, len(response.content))


def load_raw_data(path=RAW_DATA_FILE) -> pd.DataFrame:
    """Load the raw Telco churn CSV into a DataFrame, downloading it if needed.

    Returns
    -------
    pd.DataFrame
        The raw, unmodified customer churn dataset.
    """
    if not path.exists():
        download_dataset(destination=path)

    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded dataset with shape %s", df.shape)
    return df


if __name__ == "__main__":
    data = load_raw_data()
    print(data.head())
    print(f"\nShape: {data.shape}")
