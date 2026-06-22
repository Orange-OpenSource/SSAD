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
Data module for general tabular datasets using PyTorch Lightning and scikit-learn pipelines.
Handles loading, preprocessing, splitting, and dataloader creation.
"""

from pathlib import Path
from typing import Optional, List, Union, Callable
import logging
import os
import lightning as L
import pandas as pd
from torch.utils.data import DataLoader
from sklearn import set_config  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import MinMaxScaler  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]

from ssad.datamodules.dataframe_with_labels import DataFrameWithLabels
from .pipeline import create_preprocessing_pipeline
from .utils import (
    controlled_split,
    controlled_split_from_index_file,
    generate_label_index_file,
    load_rows_by_index,
    encode_labels_inplace,
)

pd.options.mode.copy_on_write = True
set_config(transform_output="pandas")
logger = logging.getLogger(__name__)


class GeneralTabularDataModule(
    L.LightningDataModule
):  # pylint: disable=too-many-instance-attributes
    """
    A LightningDataModule for tabular datasets with customizable preprocessing.
    Only destined for methods that do not take into account the temporal dependencies.

    Loads a CSV file, applies preprocessing (casting, encoding, imputation, scaling),
    splits it into train/val/test sets with contamination control, and builds PyTorch DataLoaders.

    dataset_path (str): path to the original data
    large_dataset(bool): if true, uses an indices csv file built in chunks
                         to prevent full data loading and limit RAM usage.
    label_column (str): Name of the column indicating class labels.
    p_train (float): Proportion of total data for training.
    p_val (float): Proportion of total data for validation.
    pa_train (float): Proportion of anomalies in training set.
    pa_val (float): Proportion of anomalies in validation set.
    pa_test (float): Proportion of anomalies in test set.
    n_rows (Optional[int], optional): Total number of data points to include.
        Defaults to dataset size.
        Limits the dataset size by restricting the total number of rows to n_rows if specified.
    random_state (int, optional): Seed for reproducibility.
    shuffling (bool, optional): Whether to shuffle indices before splitting train/val/test sets.


    Should contain a 0/1 valued column for anomaly labelling: 0 for normal samples, 1 for anomalies
    """

    pipeline: Pipeline

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        dataset_path: str,
        large_dataset: bool = False,
        label_column: str = "label",
        batch_size: int = 64,
        p_train: float = 0.4,
        p_val: float = 0.1,
        pa_train: float = 0.05,
        pa_val: float = 0.05,
        pa_test: float = 0.05,
        n_rows: Optional[int] = None,
        preprocessed_dir: Optional[str] = None,
        normal_labels: Union[List, int] = 0,
        anomaly_labels: Union[List, int] = 1,
        categorical_cols: Optional[List[str]] = None,
        drop_cols: Optional[List[str]] = None,
        count_encode_cols: Optional[List[str]] = None,
        scaler: Optional[Union[Callable[[], object], object]] = lambda: MinMaxScaler(
            feature_range=(0, 1)
        ),
        replace_inf: bool = True,
        # TODO change to drop NaNs
        fillna: Optional[Union[Callable[[], object], object]] = lambda: SimpleImputer(
            strategy="mean"
        ),
        transform=None,
        target_transform=None,
        random_state: int = 42,
        shuffling=False,
    ):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.large_dataset = large_dataset
        self.label_column = label_column
        self.batch_size = batch_size
        self.p_train = p_train
        self.p_val = p_val
        self.pa_train = pa_train
        self.pa_val = pa_val
        self.pa_test = pa_test
        self.n_rows = n_rows
        self.preprocessed_dir = (
            Path(preprocessed_dir)
            if preprocessed_dir is not None
            else self.dataset_path.parent.parent / "Preprocessed" / "data"
        )
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        self.normal_labels = normal_labels
        self.anomaly_labels = anomaly_labels
        self.categorical_cols = categorical_cols
        self.drop_cols = drop_cols
        self.count_encode_cols = count_encode_cols
        self.scaler = scaler
        self.replace_inf = replace_inf
        self.fillna = fillna
        self.transform = transform
        self.target_transform = target_transform
        self.random_state = random_state
        self.shuffling = shuffling

        self.pipeline = create_preprocessing_pipeline(
            categorical_cols=self.categorical_cols,
            cols_to_drop=self.drop_cols,
            count_encode_cols=self.count_encode_cols,
            scaler=self.scaler,
            replace_inf=self.replace_inf,
            fillna=self.fillna,
        )
        # paths
        base_name = f"{self.dataset_path.stem}_n={self.n_rows}_train={self.pa_train}-{self.p_train}_val={self.pa_val}-{self.p_val}_test={self.pa_test}"  # pylint: disable=line-too-long
        self.preprocessed_files = {
            "train": self.preprocessed_dir / f"{base_name}_train.csv",
            "validation": self.preprocessed_dir / f"{base_name}_val.csv",
            "test": self.preprocessed_dir / f"{base_name}_test.csv",
        }
        self.train: Optional[DataFrameWithLabels] = None
        self.val: Optional[DataFrameWithLabels] = None
        self.test: Optional[DataFrameWithLabels] = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):  # pylint: disable=unused-argument
        if all(p.exists() for p in self.preprocessed_files.values()):
            logger.info("🔄 Loading preprocessed files.")
            train = pd.read_csv(self.preprocessed_files["train"])
            val = pd.read_csv(self.preprocessed_files["validation"])
            test = pd.read_csv(self.preprocessed_files["test"])
        else:
            logger.info("🔄 Preprocessing the dataset...")
            if self.large_dataset:
                # create a csv file for indices only
                folder = os.path.dirname(self.dataset_path)
                base = os.path.splitext(os.path.basename(self.dataset_path))[0]
                path_index_file = os.path.join(folder, f"{base}_index.csv")

                # process the data by chunks to produce the single column CSV
                #   containing the anomaly labels
                generate_label_index_file(
                    self.dataset_path, self.label_column, path_index_file
                )

                # read the produced CSV, encode the labels
                data = pd.read_csv(path_index_file)
                self.encode_labels_inplace(data)

                train_idx, val_idx, test_idx = controlled_split_from_index_file(
                    df=data,
                    label_column=self.label_column,
                    p_train=self.p_train,
                    p_val=self.p_val,
                    pa_train=self.pa_train,
                    pa_val=self.pa_val,
                    pa_test=self.pa_test,
                    n_rows=self.n_rows,
                    random_state=self.random_state,
                    shuffling=self.shuffling,
                )

                train = load_rows_by_index(self.dataset_path, train_idx)
                val = load_rows_by_index(self.dataset_path, val_idx)
                test = load_rows_by_index(self.dataset_path, test_idx)
                data_cols = train.columns

            else:
                data = pd.read_csv(self.dataset_path)
                self.encode_labels_inplace(data)
                # encode_labels_inplace(data, self.normal_labels, self.anomaly_labels)

                train, val, test = controlled_split(
                    df=data,
                    label_column=self.label_column,
                    p_train=self.p_train,
                    p_val=self.p_val,
                    pa_train=self.pa_train,
                    pa_val=self.pa_val,
                    pa_test=self.pa_test,
                    n_rows=self.n_rows,
                    random_state=self.random_state,
                    shuffling=self.shuffling,
                )
                data_cols = train.columns

            train = self.pipeline.fit_transform(train)
            val = self.pipeline.transform(val)
            test = self.pipeline.transform(test)

            # Get remaining Columns
            # TODO check if necessary to define remaining columns
            remaining_cols = [
                col
                for col in data_cols
                if col not in (self.drop_cols if self.drop_cols else [])
            ]

            # Reconstruct Datasets
            train = pd.DataFrame(train, columns=remaining_cols)
            val = pd.DataFrame(val, columns=remaining_cols)
            test = pd.DataFrame(test, columns=remaining_cols)

            logger.info("✅ Dataset preprocessing completed.")
            logger.info("💾 Saving preprocessed splits to CSV...")

            # Save Datasets
            self.save_preprocessed_data(train, val, test)
            logger.info("✅ Preprocessed files successfully saved.")

        self.train = DataFrameWithLabels(
            train, self.label_column, self.transform, self.target_transform
        )
        self.val = DataFrameWithLabels(
            val, self.label_column, self.transform, self.target_transform
        )
        self.test = DataFrameWithLabels(
            test, self.label_column, self.transform, self.target_transform
        )

        logger.info("✅ Dataset Loaded.")

        train_total, train_normal, train_anomaly = self.train.get_stats()
        val_total, val_normal, val_anomaly = self.val.get_stats()
        test_total, test_normal, test_anomaly = self.test.get_stats()

        logger.info("Train Dataset :")
        logger.info("   Total samples : %s", train_total)
        logger.info(
            "   Normal samples : %s %%",
            (train_normal / train_total) * 100 if train_total > 0 else 0,
        )
        logger.info(
            "   Anormal samples : %s %%",
            (train_anomaly / train_total) * 100 if train_total > 0 else 0,
        )

        logger.info("Validation Dataset :")
        logger.info("   Total samples : %s", val_total)
        logger.info(
            "   Normal samples : %s %%",
            (val_normal / val_total) * 100 if val_total > 0 else 0,
        )
        logger.info(
            "   Anomalous samples : %s %%",
            (val_anomaly / val_total) * 100 if val_total > 0 else 0,
        )

        logger.info("Test Dataset :")
        logger.info("   Total samples : %s", test_total)
        logger.info(
            "   Normal samples : %s %%",
            (test_normal / test_total) * 100 if test_total > 0 else 0,
        )
        logger.info(
            "   Anomalous samples : %s %%",
            (test_anomaly / test_total) * 100 if test_total > 0 else 0,
        )

    def encode_labels_inplace(self, df: pd.DataFrame):
        encode_labels_inplace(
            df, self.label_column, self.normal_labels, self.anomaly_labels
        )

    def train_dataloader(self):
        if self.train is None:
            raise ValueError("Cannot create dataloader, training set is None")
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        if self.val is None:
            raise ValueError("Cannot create dataloader, validation set is None")
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        if self.test is None:
            raise ValueError("Cannot create dataloader, testing set is None")
        return DataLoader(self.test, batch_size=self.batch_size)

    def save_preprocessed_data(
        self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
    ):
        """
        Save the preprocessed files in the specified folder,
        ensuring that the directory exists.
        """
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)

        train.to_csv(self.preprocessed_files["train"], index=False, encoding="utf-8")
        val.to_csv(self.preprocessed_files["validation"], index=False, encoding="utf-8")
        test.to_csv(self.preprocessed_files["test"], index=False, encoding="utf-8")
