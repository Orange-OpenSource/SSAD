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
from typing import Optional, Protocol

import torch

from ssad.confidence_estimators.confidence_intervals_configuration import (
    ConfidenceIntervalsConfiguration,
)


class SupportsConfidenceEstimation(Protocol):
    """Protocol for confidence estimators used in a SelfSupervisionCallback."""

    configuration: Optional[ConfidenceIntervalsConfiguration]

    def estimate_confidence(self, scores_batch: torch.Tensor) -> torch.Tensor:
        """Computes the confidence score over a data batch"""
