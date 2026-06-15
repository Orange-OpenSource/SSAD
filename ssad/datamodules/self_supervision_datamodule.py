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
"""Implements a Self Supervision Data module.

This data module is designed using composition with another LightningDataModule.
It relies on these
"""

from pathlib import Path
from typing import Optional
import lightning as L
from torch.utils.data import DataLoader, Dataset

from .dataset_with_confidence import DatasetWithConfidence, init_confidence_from_csv
from .dataset_interfaces import DatasetWithLabels, DatasetWithInputDim


class SelfSupervisionDataModule(L.LightningDataModule):
    """Datamodule for Self Supervision trainings and tests.
    Wraps an original datamodule to add confidence during training.
    """

    train: DatasetWithConfidence
    input_dim: int
    batch_size: int

    # TODO parameters for training/supervising dataloader
    def __init__(
        self, datamodule: L.LightningDataModule, path_confidence: Optional[str] = None
    ) -> None:
        super().__init__()
        self.datamodule = datamodule

        if hasattr(self.datamodule, "prepare_data"):
            self.datamodule.prepare_data()

        self.setup(stage="fit")

        # Prepare data, get input_dim from prepared data,
        # adapt the datamodule to add confidence data
        #TODO fix: input dim is typed as int
        if (
            hasattr(self.datamodule, "train")
            and isinstance(self.datamodule.train, Dataset)
            and isinstance(self.datamodule.train, DatasetWithInputDim)
            and isinstance(self.datamodule.train, DatasetWithLabels)
        ):
            path = Path(path_confidence) if path_confidence is not None else None
            confidence = init_confidence_from_csv(self.datamodule.train, path)
            self.input_dim = self.datamodule.train.input_dim()
            self.train = DatasetWithConfidence(self.datamodule.train, confidence)
        else:
            raise ValueError("Invalid 'train' parameter in datamodule.")

        if hasattr(self.datamodule, "batch_size") and isinstance(
            self.datamodule.batch_size, int
        ):
            self.batch_size = self.datamodule.batch_size
        else:
            raise ValueError("Missing 'batch_size' parameter in datamodule.")

    def setup(self, stage):
        self.datamodule.setup(stage)

    def train_dataloader(self):
        # Try Adding num_workers=os.cpu_count(), pin_memory=True, persistent_workers=True
        return DataLoader(dataset=self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        #return self.datamodule.val_dataloader()
        return DataLoader(dataset=self.datamodule.val, batch_size=self.batch_size)

    def test_dataloader(self):
        # return self.datamodule.test_dataloader()
        return DataLoader(dataset=self.datamodule.test, batch_size=self.batch_size)

    def supervision_dataloader(self) -> DataLoader:
        """Dataloader used during supervision.
        Loads every sample of the dataset in order, without repetition.

        Returns:
            DataLoader: supervision dataloader.
        """
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=(
                self.train.collate_fn if hasattr(self.train, "collate_fn") else None
            ),
        )
