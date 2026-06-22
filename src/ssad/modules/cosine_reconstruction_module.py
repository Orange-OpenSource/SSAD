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
Implements a cosine reconstruction module.

Normal samples reconstructions should be collinear with the original samples.
Abnormal samples reconstructions should be orthogonal to the original samples.
"""

from typing import List
import logging
import torch

# TODO: check if we can use the masked tensor API
# from torch.masked import MaskedTensor

from .self_supervision_module import SelfSupervisionModule

logger = logging.getLogger(__name__)


class CosineReconstructionModule(SelfSupervisionModule):
    """
    Class implementing a cosine reconstruction self-supervision.
    CosineSimilarity will return a 0 value for 0-tensors.
    """

    criterion = torch.nn.CosineSimilarity(dim=1, eps=1e-5)

    def compute_loss(
        self, data: torch.Tensor, confidence: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        confidence = confidence.squeeze()

        mask_positive = confidence == 1
        mask_negative = confidence == -1

        if batch_idx == 0:
            logger.debug("mask positive shape : %s", mask_positive.shape)
            logger.debug("mask negative shape : %s", mask_negative.shape)

        loss_pos = (
            self._positive_loss(data[mask_positive], confidence[mask_positive])
            if mask_positive.any()
            else torch.tensor(0.0, device=data.device)
        )
        loss_neg = (
            self._negative_loss(data[mask_negative], confidence[mask_negative])
            if mask_negative.any()
            else torch.tensor(0.0, device=data.device)
        )

        if batch_idx == 0:
            logger.debug("loss positive shape : %s", loss_pos.shape)
            logger.debug("loss negative shape : %s", loss_neg.shape)

        total = mask_positive.sum() + mask_negative.sum()
        if total == 0:
            return torch.tensor(0.0, device=data.device)

        if batch_idx == 0:
            logger.debug("total : %s", total)
        return (loss_pos.sum() + loss_neg.sum()) / total

    def score(self, batch: torch.Tensor) -> torch.Tensor:
        return self.criterion(self.model.forward(batch), batch)

    def _positive_loss(self, data, confidence):
        score = self.score(data)
        return confidence.abs() * (1 - score) ** 2

    def _negative_loss(self, data, confidence):
        score = self.score(data)
        return confidence.abs() * score**2

    def _uncertain_loss(self, data, confidence):
        pass

    def _prediction_score(self, data):
        score = self.score(data)
        return 1 - score

    def update_thresholds(self) -> List[float]:
        """Return the decision threshold according to the selected strategy."""
        if self.threshold_strategy == "auto":
            config = getattr(self.confidence_estimator, "configuration", None)
            if config is not None:
                return [
                    1 - float(config.abnormal.right),
                    1 - float(config.normal.left),
                    self.manual_threshold,
                ]
            logger.warning(
                "'auto' strategy selected but confidence intervals are missing "
                "or invalid. Falling back to manual threshold."
            )
        return [self.manual_threshold]
