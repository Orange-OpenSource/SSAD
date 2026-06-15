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
from typing import Protocol, Any, runtime_checkable


@runtime_checkable
class DatasetWithLabels(Protocol):
    """Interface for a Dataset containing two attributes:
    data (Any) : data used by a model for its task
    labels (Any) : ground truth for the data

    """

    labels: Any
    data: Any


@runtime_checkable
class DatasetWithInputDim(Protocol):
    """Interface for a Dataset with the input_dim method
    This method is necessary to initialize the input layer of an autoencoder.
    """

    def input_dim(self) -> int:
        """Returns the dimension of the data"""

    # def collate(self, batch : List[int]) -> tuple[torch.Tensor, torch.Tensor]:
    #     """Collate function, used to transform a batch into a tensor

    #     Args:
    #         batch (List[int]): list of indexes in the batch

    #     Returns:
    #         torch.Tensor:
    #     """
