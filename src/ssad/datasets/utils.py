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
Useful functions for dataset transforms
"""

from typing import List, Sequence, Union
from typing import Optional
import logging
import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def replace_values_with_nan(df: pd.DataFrame, values: Sequence[Union[int, float]]):
    """Replaces values with NaN in a DataFrame

    Args:
        df (pd.DataFrame): dataframe to process
        values (Sequence[float | int]): Values to replace with NaN, e.g., np.inf

    Returns:
        pd.DataFrame: transformed dataframe
    """
    # deal with NaNs
    df = df.replace(to_replace=values, value=np.nan)
    return df


def to_categorical(df: pd.DataFrame, cols: List[str]):
    """Changes column types in a dataframe to 'category'

    Args:
        df (pd.DataFrame): dataframe to process
        cols (List[str]): List of column names to transform

    Returns:
        pd.DataFrame: transformed dataframe
    """
    df[cols] = df[cols].astype("category")
    return df


def fuse_cols(df: pd.DataFrame, cols: List[str], new_col_name: str):
    """Concatenates a set of columns (treated as strings) in a dataframe and saves the result
    in a new column, e.g., [-1, -1] -> [-1, -1, '-1-1']

    Args:
        df (pd.DataFrame): dataframe to process
        cols (List[str]): List of column names to concatenate
        new_col_name (str): name of the created column containing the result

    Returns:
        pd.DataFrame: transformed dataframe
    """
    df[new_col_name] = df[cols].copy().astype("string").sum(axis=1)
    return df


def drop_cols(df: pd.DataFrame, cols: List[str]):
    """Drops a set of columns from a dataframe

    Args:
        df (pd.DataFrame): dataframe to process
        cols (List[str]): _List of column names to drôp

    Returns:
        pd.DataFrame: transformed dataframe
    """
    df = df.drop(cols, axis=1)
    return df


def time_split_train_val_test(
    data: pd.DataFrame, train_percent: float, val_percent: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the dataset into training, validation and test subsets.
    Respects the time dependencies by taking:
    - the first train_percent percent data for training,
    - from train_percent to train_percent + val_percent data for validation,
    - the rest for test


    Args:
        data (dataframe): contains the time data, without labels
        train_percent (float): percentage of data used for training
        val_percent (float): percentage of data used for validation
    """

    end_index_train = int(train_percent * len(data))
    data_train = data[:end_index_train]

    if val_percent == 0:
        data_val = pd.DataFrame([])
    else:
        end_index_val = int((train_percent + val_percent) * len(data))
        data_val = data[end_index_train:end_index_val]

    data_test = data[end_index_val:]

    return (data_train, data_val, data_test)


def controlled_split(
    df: pd.DataFrame,
    label_column: str,
    p_train: float,
    p_val: float,
    pa_train: float,
    pa_val: float,
    pa_test: float,
    n_rows: Optional[int] = None,
    random_state: int = 42,
    shuffling=False,
):  # pylint: disable=too-many-locals,too-many-positional-arguments
    """
    Splits a DataFrame into training, validation, and test sets with controlled class proportions.
    Only use for time-independent methods, as temporal dependencies are not respected.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        label_column (str): Name of the column indicating class labels.
        p_train (float): Proportion of total data for training.
        p_val (float): Proportion of total data for validation.
        pa_train (float): Proportion of anomalies in training set.
        pa_val (float): Proportion of anomalies in validation set.
        pa_test (float): Proportion of anomalies in test set.
        n_rows (Optional[int], optional): Total number of rows to include. Defaults to dataset size.
            Limits the dataset size by restricting the total number of rows to n_rows if specified.
        random_state (int, optional): Random seed for shuffling. Defaults to 42.
        shuffling (bool, optional): Whether to shuffle data before splitting. Defaults to False.

    Returns:
        data (Tuple) : A tuple containing three elements:
                - train (pd.DataFrame): train set.
                - val (pd.DataFrame): validation set.
                - test (pd.DataFrame):  test set.
    """
    if shuffling:
        df = shuffle(df, random_state=random_state)

    # We assume beforehand that anomalies are encoded as 1 and normal instances as 0.
    anomalies = df[df[label_column] == 1]
    normals = df[df[label_column] == 0]

    available_anomalies = len(anomalies)
    available_normals = len(normals)

    # Dataset shape
    total_requested = min(
        n_rows if n_rows is not None else available_anomalies + available_normals,
        available_anomalies + available_normals,
    )

    # Total Splits
    n_train = int(total_requested * p_train)
    n_val = int(total_requested * p_val)
    n_test = total_requested - n_train - n_val

    # Anomalies per split
    m_train = int(pa_train * n_train)
    m_val = int(pa_val * n_val)
    m_test = int(pa_test * n_test)

    total_anomalies_needed = m_train + m_val + m_test
    total_normals_needed = total_requested - total_anomalies_needed

    # Logging
    anomalies_str = f"{m_train:,}/{m_val:,}/{m_test:,}"
    normals_str = f"{n_train - m_train:,}/{n_val - m_val:,}/{n_test - m_test:,}"
    logger.info("ℹ️ Requested split:")
    logger.info("Anomalies train/val/test: %s", anomalies_str)
    logger.info("Normals   train/val/test: %s", normals_str)

    # Raise Error if not enough anomalies or normal samples
    if total_anomalies_needed > available_anomalies:
        formatted_needed_anomalies = f"{total_anomalies_needed:,}"
        formatted_available_anomalies = f"{available_anomalies:,}"
        logger.error(
            "❌ Requested %s anomalies, but only %s are available.",
            formatted_needed_anomalies,
            formatted_available_anomalies,
        )
        raise ValueError("Requested anomaly count exceeds available data.")

    if total_normals_needed > available_normals:
        formatted_needed_normals = f"{total_normals_needed:,}"
        formatted_available_normals = f"{available_normals:,}"
        logger.error(
            "❌ Requested %s normal samples, but only %s are available.",
            formatted_needed_normals,
            formatted_available_normals,
        )
        raise ValueError("Requested normal sample count exceeds available data.")

    # Final Selection
    anomalies_train = anomalies.iloc[:m_train]
    anomalies_val = anomalies.iloc[m_train : m_train + m_val]
    anomalies_test = anomalies.iloc[m_train + m_val : m_train + m_val + m_test]

    normals_train = normals.iloc[: n_train - m_train]
    normals_val = normals.iloc[n_train - m_train : n_train - m_train + n_val - m_val]
    normals_test = normals.iloc[
        n_train
        - m_train
        + n_val
        - m_val : n_train
        - m_train
        + n_val
        - m_val
        + n_test
        - m_test
    ]

    train = pd.concat([anomalies_train, normals_train])
    val = pd.concat([anomalies_val, normals_val])
    test = pd.concat([anomalies_test, normals_test])

    return train, val, test


def controlled_split_from_index_file(
    df: pd.DataFrame,
    label_column: str,
    p_train: float,
    p_val: float,
    pa_train: float,
    pa_val: float,
    pa_test: float,
    n_rows: Optional[int] = None,
    random_state: int = 42,
    shuffling=False,
):  # pylint: disable=too-many-locals,too-many-positional-arguments
    """
    Generates train, validation, and test index lists from an index CSV file
    with controlled class proportions.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
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
        shuffling (bool, optional): Whether to shuffle indices before splitting.

    Returns:
        data_idx (Tuple) : A tuple containing three elements:
                - train_idx (array-like): Indices for the training set.
                - val_idx (array-like): Indices for the validation set.
                - test_idx (array-like): Indices for the test set.
    """

    if shuffling:
        df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    anomalies = df[df[label_column] == 1]
    normals = df[df[label_column] == 0]

    # compute available number of anomalous / normal samples
    available_anomalies = len(anomalies)
    available_normals = len(normals)
    total_requested = min(
        n_rows if n_rows is not None else available_anomalies + available_normals,
        available_anomalies + available_normals,
    )

    # compute the number of samples in each set, using the given proportions
    n_train = int(total_requested * p_train)
    n_val = int(total_requested * p_val)
    n_test = total_requested - n_train - n_val

    # compute the number of required anomalies in each set
    m_train = int(pa_train * n_train)
    m_val = int(pa_val * n_val)
    m_test = int(pa_test * n_test)

    total_anomalies_needed = m_train + m_val + m_test
    total_normals_needed = total_requested - total_anomalies_needed

    # Logging
    anomalies_str = f"{m_train:,}/{m_val:,}/{m_test:,}"
    normals_str = f"{n_train - m_train:,}/{n_val - m_val:,}/{n_test - m_test:,}"
    logger.info("ℹ️ Requested split:")
    logger.info("Anomalies train/val/test: %s", anomalies_str)
    logger.info("Normals   train/val/test: %s", normals_str)

    # Raise Error if not enough anomalies or normal samples
    if total_anomalies_needed > available_anomalies:
        formatted_needed_anomalies = f"{total_anomalies_needed:,}"
        formatted_available_anomalies = f"{available_anomalies:,}"
        logger.error(
            "❌ Requested %s anomalies, but only %s are available.",
            formatted_needed_anomalies,
            formatted_available_anomalies,
        )
        raise ValueError("Requested anomaly count exceeds available data.")

    if total_normals_needed > available_normals:
        formatted_needed_normals = f"{total_normals_needed:,}"
        formatted_available_normals = f"{available_normals:,}"
        logger.error(
            "❌ Requested %s normal samples, but only %s are available.",
            formatted_needed_normals,
            formatted_available_normals,
        )
        raise ValueError("Requested normal sample count exceeds available data.")

    anomalies_idx = anomalies["index"].values
    normals_idx = normals["index"].values

    anomalies_train = anomalies_idx[:m_train]
    anomalies_val = anomalies_idx[m_train : m_train + m_val]
    anomalies_test = anomalies_idx[m_train + m_val : m_train + m_val + m_test]

    normals_train = normals_idx[: n_train - m_train]
    normals_val = normals_idx[n_train - m_train : n_train - m_train + n_val - m_val]
    normals_test = normals_idx[
        n_train
        - m_train
        + n_val
        - m_val : n_train
        - m_train
        + n_val
        - m_val
        + n_test
        - m_test
    ]

    train_idx = list(anomalies_train) + list(normals_train)
    val_idx = list(anomalies_val) + list(normals_val)
    test_idx = list(anomalies_test) + list(normals_test)

    return train_idx, val_idx, test_idx


def generate_label_index_file(
    path_csv, label_column, path_index_file, chunksize=100_000
):
    """
    Creates an index file mapping data indices to label values from a CSV file.
    The function reads the CSV file in chunks of size `chunksize` to handle large files efficiently.
    This approach prevents loading the entire file into memory at once, making the process scalable
    for very large datasets.
    """
    if os.path.exists(path_index_file):
        logger.info("✅ Index file already exists at %s", path_index_file)
        return

    logger.info("🔄 Creating index file at %s...", path_index_file)
    indices = []
    for chunk in pd.read_csv(path_csv, chunksize=chunksize, usecols=[label_column]):
        idx_start = chunk.index.start if hasattr(chunk.index, "start") else 0
        indices.append(
            pd.DataFrame(
                {
                    "index": range(idx_start, idx_start + len(chunk)),
                    label_column: chunk[label_column].values,
                }
            )
        )
    index_df = pd.concat(indices, ignore_index=True)
    index_df.to_csv(path_index_file, index=False)
    logger.info("✅ Index file created at %s", path_index_file)


def load_rows_by_index(path_csv, target_indices, chunksize=100_000):
    """
    Only Load rows from the CSV whose indices are in `target_indices`.
    """
    target_set = set(target_indices)
    rows = []
    current_index = 0

    for chunk in pd.read_csv(path_csv, chunksize=chunksize):
        chunk_indices = range(current_index, current_index + len(chunk))
        mask = [i in target_set for i in chunk_indices]
        selected = chunk[mask]
        rows.append(selected)
        current_index += len(chunk)

    df = pd.concat(rows, ignore_index=True)
    return df


def encode_labels_inplace(
    df: pd.DataFrame,
    label_column: str,
    normal_labels: Union[List, int],
    anomaly_labels: Union[List, int],
):
    """
    Encode label values in the DataFrame's 'label' column to 0 for normal and 1 for anomalies.
    This function modifies the DataFrame in place.

    Args:
        df (pd.DataFrame): The DataFrame containing a 'label' column.
        normal_labels (Union[List, int]): List of values or a single value
                                          representing normal samples.
        anomaly_labels (Union[List, int]): List of values or a single value
                                           representing anomalous samples.
    """
    # Convert to list if a single int is provided
    if isinstance(normal_labels, int):
        normal_labels = [normal_labels]
    if isinstance(anomaly_labels, int):
        anomaly_labels = [anomaly_labels]

    df[label_column] = df[label_column].apply(
        lambda x: 0 if x in normal_labels else (1 if x in anomaly_labels else None)
    )
