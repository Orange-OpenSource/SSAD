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
"""Implements the base class for supervisors.
A supervisor is in charge of maintaining confidence scores during training
for each training sample.
"""

from typing import Protocol, Optional

import matplotlib.figure
import torch

from ssad.confidence_estimators.confidence_intervals_configuration import (
    ConfidenceIntervalsConfiguration,
)


class SupportsDistributionAnalysis(Protocol):
    """Interface for a supervisor.
    Implements the global supervisor logic:
    During training, the confidence estimator is used to compute a confidence score
    for each training sample. This confidence will determine which error function to
    apply for each sample.
    The confidence estimator is periodically recalibrated after analyzing a
    distribution of training samples scores.

    """

    @torch.no_grad()
    def analyze_distribution(
        self,
        scores_batch: torch.Tensor,
        current_conf: Optional[ConfidenceIntervalsConfiguration],
    ) -> ConfidenceIntervalsConfiguration:
        """Extracts a Confidence intervals configuration from the analysis of a distribution.

        Args:
            distribution (torch.Tensor): distribution of scores to analyze

        Returns:
            ConfidenceIntervalsConfiguration: new configuration
        """

    def plot_analysis(
        self, style="default", plot_size=(16, 22)
    ) -> matplotlib.figure.Figure:
        """plots a figure visualizing the supervising distribution.

        Args:
            style (str, optional): matplotlib style for plot. Defaults to "default".
            plot_size (tuple, optional): size of plot. Defaults to (16,22).

        Returns:
            matplotlib.figure.Figure: Figure
        """
