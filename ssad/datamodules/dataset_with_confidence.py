# Software Name : Self-Supervised Anomaly Detection
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html
#
# Authors: see CONTRIBUTORS
# Software description: A Python library for autoencoder-based anomaly detection
#          based on self-supervised training with dynamic sample confidence updates.
"""
Defines the torch.Dataset specialization for
pd.Datasets with a 'label' column
"""

from collections.abc import Sized
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from .dataset_interfaces import DatasetWithLabels

CONFIDENCE_COLUMN_NAME = "confidence"


class DatasetWithConfidence(Dataset):
    """
    Class that derives from torch.Dataset.
    Defines the required methods for a dataset that
    is formed of a pd.DataFrame with a column that corresponds
    to the row label.
    """

    def __init__(self, dataset: DatasetWithLabels, confidence: torch.Tensor):

        if not isinstance(dataset, Sized) or len(dataset) != confidence.size(dim=0):
            raise ValueError("Size mismatch between dataset and confidence tensor.")

        self.dataset = dataset
        self.confidence = confidence

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        # TODO check if self.dataset[idx] is a tuple or a list
        # in case there is no label
        data = self.dataset[idx]
        return {
            "data": data[0],
            "label": data[1],
            "confidence": self.confidence[idx],
        }


def init_confidence_from_csv(dataset: Dataset, path: Optional[Path]) -> torch.Tensor:
    """Loads a dataframe with confidence scores for each sample of the dataframe.

    Args:
        data (pd.Dataframe): dataframe
        path (Path): path to csv file with a single column containing the confidence scores.

    Returns:
        confidence (torch.Tensor): tensor with initial confidence values.
    """

    if not isinstance(dataset, Sized):
        raise TypeError("Provided dataset does not support the len() method.")

    if path is not None:
        confidence_df = pd.read_csv(path)

        if CONFIDENCE_COLUMN_NAME not in confidence_df:
            raise ValueError(f"Missing column: {CONFIDENCE_COLUMN_NAME}")

        confidence = torch.from_numpy(confidence_df[CONFIDENCE_COLUMN_NAME].values)
    else:
        confidence = torch.ones(len(dataset))
    return confidence


def save_confidence_to_csv(confidence_dataset: DatasetWithConfidence, path: Path):
    """Saves confidence scores to csv file.
    Uses a dataloader to iterate over the dataset and extract confidence scores
    for each sample into a list.
    This list is then saved to a csv file by converting it to a pd.DataFrame.

    Args:
        confidence_dataset (DatasetWithConfidence): dataset with confidence scores to save.
        path (Path): path to csv file.
    """
    confidence = confidence_dataset.confidence.detach().cpu().numpy()

    confidence_df = pd.DataFrame(
        confidence,
        columns=[CONFIDENCE_COLUMN_NAME],
    )
    confidence_df.to_csv(path)
