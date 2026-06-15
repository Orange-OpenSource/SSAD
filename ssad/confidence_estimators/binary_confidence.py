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
Implements a binary confidence estimator.
A sample is either normal or abnormal, otherwise it is omitted (zero confidence).
"""

import torch
from .confidence_estimator import BaseConfidenceEstimator


class BinaryConfidence(BaseConfidenceEstimator):
    """Binary confidence estimator"""

    def _confidence_normal(self, score):
        return torch.ones_like(score)

    def _confidence_abnormal(self, score):
        return torch.full_like(score, -1)

    def _confidence_unknown(self, score):
        return torch.zeros_like(score)
