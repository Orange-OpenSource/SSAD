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
Provides the base class for confidence estimators.
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch

from .confidence_intervals_configuration import (
    ConfidenceIntervalsConfiguration,
)
from .supports_confidence_estimation import SupportsConfidenceEstimation


class BaseConfidenceEstimator(ABC, SupportsConfidenceEstimation):
    """Base class for confidence estimators.
    A confidence estimator relies on four intervals with associated scoring functions
    to provide a confidence score given a criterion score for a sample.

    The criterion score can be for instance a reconstruction score, or the norm of
    the gradient of the reconstruction error.

    The intervals define the domain for model scores for the four different confidence behaviors:
    - normal: samples with model scores in this interval are considered as normal
    - abnormal: samples with model scores in this interval are considered as abnormal
    - unknown_positive: samples with model scores in this interval are
        considered as unknown, but leaning towards a normal sample.
    - unknown_negative: samples with model scores in this interval are
        considered as unknown, but leaning towards an abnormal sample.

    Each interval is associated with a "criterion score to confidence score" conversion function.
    These functions should be implemented in the _estimate_confidence_from_model_score method.
    """

    def __init__(self):
        super().__init__()
        self.configuration: Optional[ConfidenceIntervalsConfiguration] = None
        self.distribution: Optional[torch.Tensor] = None

    @abstractmethod
    def _confidence_normal(self, score):
        raise NotImplementedError()

    @abstractmethod
    def _confidence_abnormal(self, score) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def _confidence_unknown(self, score) -> torch.Tensor:
        raise NotImplementedError()

    @torch.no_grad()
    def estimate_confidence(self, scores_batch: torch.Tensor) -> torch.Tensor:
        """Estimates the confidence in a batch by retrieving the criterion score
        and translating into a confidence score.

        Args:
            scores_batch (torch.Tensor): batch whose confidence is to be estimated.

        Returns:
            torch.Tensor: confidence score
        """
        # TODO: check this order of computation
        confidence = self._confidence_unknown(scores_batch)

        if self.configuration is None:
            raise ValueError("Confidence estimator configuration is None")

        # TODO: rework signatures of confidence normal/abnormal/unknown
        normal_confidences = self._confidence_normal(scores_batch)
        abnormal_confidences = self._confidence_abnormal(scores_batch)

        confidence = torch.where(
            self.configuration.normal.contains_tensor_mask(scores_batch),
            normal_confidences,
            confidence,
        )
        confidence = torch.where(
            self.configuration.abnormal.contains_tensor_mask(scores_batch),
            abnormal_confidences,
            confidence,
        )

        return confidence
