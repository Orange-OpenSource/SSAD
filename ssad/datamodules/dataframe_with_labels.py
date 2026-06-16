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

import torch
from torch.utils.data import Dataset


class DataFrameWithLabels(Dataset):
    """
    Class that derives from torch.Dataset.
    Defines the required methods for a dataset that
    is formed of a pd.DataFrame with a column that corresponds
    to the row label.
    """

    def __init__(
        self,
        data,
        label_column_name: str,
        transform=None,
        target_transform=None,
    ):
        super().__init__()
        labels = data[label_column_name]
        features = data.drop(columns=[label_column_name])

        self.data = torch.tensor(features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx].unsqueeze(0)

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label

    def input_dim(self):
        """Returns the size of the samples of a dataset

        Args:
            dataset (Dataset): dataset whose row size is required

        Returns:
            int: length of the dataset samples
        """
        # return len(self.data.columns)
        return self.data.shape[1]

    def collate(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate function, used to transform a batch into a tensor

        Args:
            batch (List[int]): list of indexes in the batch

        Returns:
            torch.Tensor:
        """
        data, labels = zip(*batch)
        data_tensor = torch.stack(data)
        labels_tensor = torch.stack(labels)
        return data_tensor, labels_tensor

    def get_stats(self) -> tuple:
        """
        Returns the total number of samples, number of normal samples, and number of anomalies.
        Assumes that labels are 0 for normal and 1 for anomalies.

        Returns:
            tuple: (total, normal, anomaly)
        """
        total = len(self.labels)
        normal = int((self.labels == 0).sum().item())
        anomaly = int((self.labels == 1).sum().item())
        return total, normal, anomaly
