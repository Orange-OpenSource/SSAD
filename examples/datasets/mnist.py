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
MNIST Anomaly Detection DataModule for PyTorch Lightning.
Converts MNIST images to tabular format and applies contamination-controlled splitting.
"""
from typing import Optional, List, Union, Callable
import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms # type: ignore[import-untyped]
from sklearn.preprocessing import MinMaxScaler # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer # type: ignore[import-untyped]

import ssad

PATH_DATASETS = os.environ.get(
    "PATH_DATASETS", f"{Path(__file__).resolve().parent}/data"
)
logger = logging.getLogger(__name__)


class MNISTAnomalyDataModule(ssad.GeneralTabularDataModule):
    """
    Lightning DataModule for MNIST anomaly detection using tabular preprocessing.
    Converts MNIST images to flattened pixel data and applies contamination control.
    """
    def __init__(
        self,
        dataset_path: str = f"{PATH_DATASETS}/MNIST/MNIST_original",
        large_dataset: bool = False,
        label_column: str = "label",
        batch_size: int = 64,
        p_train: float = 0.4,
        p_val: float = 0.1,
        pa_train: float = 0.05,
        pa_val: float = 0.05,
        pa_test: float = 0.05,
        n_rows: Optional[int] = None,
        preprocessed_dir: Optional[str] = f"{PATH_DATASETS}/MNIST/MNIST_processed",
        normal_labels: Union[List, int] = 4,
        anomaly_labels: Union[List, int] = [i for i in range(10) if i != 4],
        categorical_cols: Optional[List[str]] = None,
        drop_cols: Optional[List[str]] = None,
        count_encode_cols: Optional[List[str]] = None,
        scaler: Optional[Union[Callable[[], object], object]] = lambda: MinMaxScaler(feature_range=(0, 1)), # pylint: disable=line-too-long
        replace_inf: bool = True,
        fillna: Optional[Union[Callable[[], object], object]] = lambda: SimpleImputer(strategy="mean"), # pylint: disable=line-too-long
        transform=None,
        target_transform=None,
        random_state: int = 42,
        shuffling = False
    ):
        """
        Initialize MNIST anomaly detection data module.
        
        Args:
            dataset_path: Path to store raw MNIST data
            normal_labels: Digit(s) to treat as normal (default: 4)
            anomaly_labels: Digit(s) to treat as anomalies (default: all except 4)
            **kwargs: Additional arguments passed to GeneralTabularDataModule
        """
        # Set up paths
        self.raw_dataset_path = Path(dataset_path)
        self.combined_path = self.raw_dataset_path.parent / "combined_data"
        self.csv_path = self.combined_path / "mnist_flattened.csv"

        # Set default preprocessed_dir if not provided
        if preprocessed_dir is None:
            preprocessed_dir = f"{PATH_DATASETS}/MNIST/MNIST_processed"

        # Initialize parent class with the CSV path
        super().__init__(
            dataset_path=str(self.csv_path),
            large_dataset=large_dataset,
            label_column=label_column,
            batch_size=batch_size,
            p_train=p_train,
            p_val=p_val,
            pa_train=pa_train,
            pa_val=pa_val,
            pa_test=pa_test,
            n_rows=n_rows,
            preprocessed_dir=preprocessed_dir,
            normal_labels=normal_labels,
            anomaly_labels=anomaly_labels,
            categorical_cols=categorical_cols,
            drop_cols=drop_cols,
            count_encode_cols=count_encode_cols,
            scaler=scaler,
            replace_inf=replace_inf,
            fillna=fillna,
            transform=transform,
            target_transform=target_transform,
            random_state=random_state,
            shuffling=shuffling
        )

    def prepare_data(self):
        """
        Download MNIST dataset, convert to flattened CSV format, and prepare for tabular processing.
        """
        # Create directories
        self.raw_dataset_path.mkdir(parents=True, exist_ok=True)
        self.combined_path.mkdir(parents=True, exist_ok=True)

        # Check if CSV already exists
        if self.csv_path.exists():
            logger.info("✅ MNIST CSV dataset already exists.")
            return

        logger.info("🔄 Preparing MNIST dataset...")

        # Check if MNIST tensors exist
        mnist_files = [
            self.raw_dataset_path / 'MNIST' / 'processed' / 'training.pt',
            self.raw_dataset_path / 'MNIST' / 'processed' / 'test.pt'
        ]

        if all(f.exists() for f in mnist_files):
            logger.info("✅ MNIST tensor dataset exists.")
        else:
            logger.info("📥 MNIST dataset not found. Downloading...")
            # Download MNIST dataset
            transform = transforms.ToTensor()
            datasets.MNIST(
                root=self.raw_dataset_path,
                train=True,
                download=True,
                transform=transform
            )
            datasets.MNIST(
                root=self.raw_dataset_path,
                train=False,
                download=True,
                transform=transform
            )

        # Load datasets
        train_dataset = datasets.MNIST(
            root=self.raw_dataset_path,
            train=True,
            download=False,
            transform=transforms.ToTensor()
        )
        test_dataset = datasets.MNIST(
            root=self.raw_dataset_path,
            train=False,
            download=False,
            transform=transforms.ToTensor()
        )

        # Convert to flattened CSV format
        logger.info("🔄 Converting MNIST to tabular format...")

        all_data = []
        all_labels = []

        # Process training data
        for data, label in train_dataset:
            flattened = data.flatten().numpy()
            all_data.append(flattened)
            all_labels.append(label)

        # Process test data
        for data, label in test_dataset:
            flattened = data.flatten().numpy()
            all_data.append(flattened)
            all_labels.append(label)

        # Create DataFrame
        all_data = np.array(all_data)
        all_labels = np.array(all_labels)

        # Create column names for flattened pixels
        pixel_cols = [f"pixel_{i}" for i in range(784)]  # 28x28 = 784 pixels

        # Create DataFrame
        df = pd.DataFrame(all_data, columns=pixel_cols)
        df[self.label_column] = all_labels

        # Save to CSV
        logger.info("💾 Saving flattened MNIST dataset to CSV...")
        df.to_csv(self.csv_path, index=False)

        logger.info("✅ MNIST dataset converted and saved to %s", self.csv_path)
        logger.info("📊 Dataset shape: %s", df.shape)
        logger.info("📊 Label distribution:\n %s", df[self.label_column].value_counts().sort_index())

    def setup(self, stage=None):
        """
        Call parent setup method after ensuring data is prepared.
        """
        # Ensure data is prepared before setup
        if not self.csv_path.exists():
            self.prepare_data()

        # Call parent setup
        super().setup(stage)

        # Log additional MNIST-specific information
        if stage is None or stage == "fit":
            logger.info("🔢 MNIST Anomaly Detection Setup:")
            logger.info("   Normal digit: %s", self.normal_labels)
            logger.info("   Anomaly digits: %s", self.anomaly_labels)
            logger.info("   Image dimensions: 28x28 (784 features)")

    def visualize_sample(self, dataset: str = "train", index: int = 0):
        """
        Visualize a sample from the dataset as an image.
        
        Args:
            dataset: Dataset to sample from ("train", "val", or "test")
            index: Index of the sample to visualize
        """
        # Get the appropriate dataset
        if dataset == "train":
            data = self.train
        elif dataset == "val":
            data = self.val
        elif dataset == "test":
            data = self.test
        else:
            raise ValueError("dataset must be 'train', 'val', or 'test'")

        if data is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")

        # Get sample
        sample, label = data[index]

        # Convert to numpy if tensor
        if torch.is_tensor(sample):
            sample = sample.numpy()

        # Reshape flattened pixels back to 28x28
        image = sample.reshape(28, 28)

        # Create plot
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {label} ({'Normal' if label == 0 else 'Anomaly'})")
        plt.axis('off')
        plt.show()

        print(f"Sample {index} from {dataset} dataset:")
        print(f"Label: {label} ({'Normal' if label == 0 else 'Anomaly'})")
        print(f"Original digit: {self._get_original_digit(label)}")
        print(f"Image shape: {image.shape}")
        print(f"Pixel value range: [{sample.min():.3f}, {sample.max():.3f}]")

    def _get_original_digit(self, encoded_label: int) -> int:
        """
        Get the original MNIST digit from encoded label.
        
        Args:
            encoded_label: Encoded label (0 for normal, 1 for anomaly)
            
        Returns:
            Original MNIST digit (0-9)
        """
        # This is a simplified approach - in practice, you'd need to track
        # the mapping during encoding or store additional metadata
        if encoded_label == 0:  # Normal
            return self.normal_labels if isinstance(self.normal_labels, int) else self.normal_labels[0]
        else:  # Anomaly
            return self.anomaly_labels if isinstance(self.anomaly_labels, int) else self.anomaly_labels[0]

