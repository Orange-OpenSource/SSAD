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
Provides a customizable preprocessing pipeline for tabular data.
Includes handling for categorical casting, column dropping, encoding, inf/nan replacement and scaling. # pylint: disable=line-too-long
"""

from typing import List, Optional, Union, Callable
import numpy as np
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
import category_encoders as ce  # type: ignore[import-untyped]
from .utils import drop_cols, replace_values_with_nan, to_categorical


def create_preprocessing_pipeline(  # pylint: disable=too-many-positional-arguments
    categorical_cols: Optional[List[str]] = None,
    cols_to_drop: Optional[List[str]] = None,
    count_encode_cols: Optional[List[str]] = None,
    scaler: Optional[Union[Callable[[], object], object]] = lambda: MinMaxScaler(
        feature_range=(0, 1)
    ),
    replace_inf: bool = True,
    fillna: Optional[Union[Callable[[], object], object]] = lambda: SimpleImputer(
        strategy="mean"
    ),
) -> Pipeline:
    """
    Creates a scikit-learn Pipeline with optional preprocessing steps for tabular data.

    Args:
        categorical_cols (Optional[List[str]]): Columns to cast as categorical.
        cols_to_drop (Optional[List[str]]): Columns to drop from the dataset.
        count_encode_cols (Optional[List[str]]): Columns to apply count encoding on.
        scaler (Optional[Callable or object]): Scaler to normalize features (default: MinMaxScaler).
        replace_inf (bool): Whether to replace inf/-inf with NaN (default: True).
        fillna (Optional[Callable or object]): Imputer for missing values (default: SimpleImputer with mean). # pylint: disable=line-too-long

    Returns:
        Pipeline: Configured sklearn Pipeline for preprocessing.
    """

    steps = []

    if categorical_cols:
        steps.append(
            (
                "cast_categorical",
                FunctionTransformer(to_categorical, kw_args={"cols": categorical_cols}),
            )
        )

    if cols_to_drop:
        steps.append(
            (
                "drop_columns",
                FunctionTransformer(drop_cols, kw_args={"cols": cols_to_drop}),
            )
        )

    if count_encode_cols:
        steps.append(
            (
                "count_encode",
                ce.CountEncoder(
                    handle_unknown=0, return_df=True, cols=count_encode_cols
                ),
            )
        )

    if replace_inf:
        steps.append(
            (
                "replace_inf_with_nan",
                FunctionTransformer(
                    func=replace_values_with_nan, kw_args={"values": [np.inf, -np.inf]}
                ),
            )
        )

    if fillna:
        imputer = fillna() if callable(fillna) else fillna
        steps.append(("fillna", imputer))

    if scaler:
        scaler_instance = scaler() if callable(scaler) else scaler
        steps.append(("scaler", scaler_instance))

    return Pipeline(steps=steps)
