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
import torch
import pandas as pd
import numpy as np

class DataFrameToTensor():
    """Convert a dataframe dataset to a tensor one.
    Args:
        df (pd.DataFrame) : dataframe to convert
        
    Returns:
        torch.Tensor : tensor equivalent to the dataframe 
    """

    # no init, no use for parameters

    def __call__(self, df) -> torch.Tensor :
        if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
            return torch.from_numpy(df.values).float()
        elif isinstance(df, np.ndarray):
            return torch.from_numpy(df).float()
        elif isinstance(df, torch.Tensor):
            return df.float()  # au cas où ce serait en double ou long
        else:
            raise TypeError(f"Type non supporté : {type(df)}")
