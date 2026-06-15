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
from typing import List, Optional

import torch

from ssad.datamodules.dataset_with_confidence import DatasetWithConfidence
from .confidence_estimator import BaseConfidenceEstimator, SupportsConfidenceEstimation


class HybridEstimator(SupportsConfidenceEstimation):

    def __init__(
        self,
        training_dataset: DatasetWithConfidence,
        estimators: List[BaseConfidenceEstimator],
        weights: Optional[List[float]] = None,
    ):

        if estimators is None:
            raise ValueError(
                "Empty estimator list during HybridEstimator construction."
            )

        if weights is None:
            self.weights = torch.ones(len(estimators)) / len(estimators)
        else:
            self.weights = torch.FloatTensor(weights)

        if len(self.weights) != len(estimators):
            raise ValueError("Different number of weights and confidence estimators")

        self.training_dataset = training_dataset
        self.estimators = estimators

    def update_training_confidence(self):
        confidence = torch.FloatTensor(torch.zeros(len(self.training_dataset)))
        for idx, estimator in enumerate(self.estimators):
            confidence_estimator = estimator.batch_confidence()
            confidence += self.weights[idx] * confidence_estimator
        return confidence
