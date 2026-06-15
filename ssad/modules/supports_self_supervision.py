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
Specifies the interface of a (Lightning) module that supports self-supervision.
"""

from typing import Protocol

import torch

from ssad.confidence_estimators.supports_confidence_estimation import (
    SupportsConfidenceEstimation,
)
from ssad.distribution_analyzers.supports_distribution_analysis import (
    SupportsDistributionAnalysis,
)


class SupportsSelfSupervision(Protocol):
    """_summary_

    Args:
        Protocol (_type_): _description_
    """

    model: torch.nn.Module
    every_n_epochs: int
    confidence_estimator: SupportsConfidenceEstimation
    distr_analyzer: SupportsDistributionAnalysis

    def score(self, batch: torch.Tensor) -> torch.Tensor:
        """Computes the sample score according to the defined criterion.

        Args:
            batch (torch.Tensor): batch to forward

        Returns:
            torch.Tensor: 1D-tensor
        """
