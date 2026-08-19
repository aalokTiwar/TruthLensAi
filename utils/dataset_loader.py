"""
Utilities for loading TruthLens AI evidence data.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "sample_dataset.json"


def load_dataset() -> List[Dict[str, Any]]:
    """
    Load the local evidence dataset.

    Returns:
        A list of evidence records.
    """

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Dataset must contain a JSON list.")

    return data