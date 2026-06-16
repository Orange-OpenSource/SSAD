"""
Utilitary functions for SSAD examples
"""

from datetime import datetime
import pytz


def run_name(dataset_name: str):
    """generates a run name given a dataset name

    Args:
        dataset_name (str): name of the dataset

    Returns:
        str: name of the run, concatenation of the dataset name and the
    """
    return (
        dataset_name
        + " "
        + datetime.now(pytz.timezone("UTC")).replace(microsecond=0).isoformat()
    )
