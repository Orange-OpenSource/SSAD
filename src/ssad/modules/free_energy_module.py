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
Implements a free energy scoring module.

Normal samples should have low reconstruction error and KL divergence.
Abnormal samples are expected to produce higher free energy scores.
"""

from typing import Callable, List
import logging
import torch
from torch import nn
from ssad.confidence_estimators.supports_confidence_estimation import (
    SupportsConfidenceEstimation,
)
from ssad.distribution_analyzers.supports_distribution_analysis import (
    SupportsDistributionAnalysis,
)
from .self_supervision_module import SelfSupervisionModule

logger = logging.getLogger(__name__)


class FreeEnergyScoringModule(SelfSupervisionModule):
    """
    Implements a free energy scoring module for VAE.
    Free energy is computed as:
        F(x) = reconstruction_error(x, x_hat) + KL_divergence
    """

    def __init__( # pylint: disable=too-many-positional-arguments
        self,
        model: nn.Module,
        m: int,
        every_n_epochs: int,
        confidence_estimator: SupportsConfidenceEstimation,
        distr_analyzer: SupportsDistributionAnalysis,
        threshold_strategy: str = "auto",  # "manual" or "auto"
        manual_threshold: float = 0.5,
    ):
        super().__init__(
            model,
            every_n_epochs,
            confidence_estimator,
            distr_analyzer,
            threshold_strategy,
            manual_threshold,
        )
        self.reconstruction_loss = torch.nn.MSELoss(reduction="none")
        self.m = m
        self.counter = 0

    @property
    def criterion(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        def compute_free_energy(
            recon_data: torch.Tensor, kl_div: torch.Tensor, data: torch.Tensor
        ) -> torch.Tensor:
            """
            Computes the free energy of each sample in the batch.

            Args:
                recon_data (Tensor): Reconstruction of input (B, D)
                kl_div (Tensor): KL divergence term (B,)
                data (Tensor): Original input (B, D)

            Returns:
                Tensor: Free energy score (B,)
            """
            recon_error = (
                self.reconstruction_loss(recon_data, data)
                .view(data.size(0), -1)
                .sum(dim=1)
            )

            free_energy = recon_error + kl_div
            # kl_scalar   = kl_div.mean()          # scalaire, gradients OK
            # if kl_scalar.item() < 0 :
            #     logger.warning("KL Div Mean is Negative")
            # free_energy = recon_error + kl_scalar  # broadcast → [B]

            if self.counter % 50 == 0:
                logger.info("data size : %s", data.size(0))
                logger.info("MIN reconstruction Error: %s", recon_error.min().item())
                logger.info("MIN KL Divergence : %s", kl_div.min().item())
                logger.info("MIN Free Energy: %s", free_energy.min().item())
            self.counter += 1

            if free_energy.min().item() < 0:
                num_negatives = (free_energy < 0).sum().item()
                percent_neg = num_negatives * 100 / free_energy.numel()
                logger.warning(
                    "Free Energy: %s%% of items are negative (%s out of %s)",
                    percent_neg,
                    num_negatives,
                    free_energy.numel(),
                )

            if recon_error.min().item() < 0:
                num_negatives_recon = (recon_error < 0).sum().item()
                percent_neg_recon = num_negatives_recon * 100 / recon_error.numel()
                logger.warning(
                    "Recon Error: %s%% of items are negative (%s out of %s)",
                    percent_neg_recon,
                    num_negatives_recon,
                    recon_error.numel(),
                )

            if kl_div.min().item() < 0:
                num_negatives_kl = (kl_div < 0).sum().item()
                percent_neg_kl = num_negatives_kl * 100 / kl_div.numel()
                logger.warning(
                    "KL Div: %s%% of items are negative (%s out of %s)",
                    percent_neg_kl,
                    num_negatives_kl,
                    kl_div.numel(),
                )

            return free_energy

        return compute_free_energy

    def compute_loss(
        self, data: torch.Tensor, confidence: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        confidence = confidence.squeeze()

        mask_positive = confidence == 1
        mask_negative = confidence == -1
        mask_uncertain = confidence == 0

        loss_pos = (
            self._positive_loss(data[mask_positive], confidence[mask_positive])
            if mask_positive.any()
            else torch.tensor([], device=data.device)
        )
        loss_neg = (
            self._negative_loss(data[mask_negative], confidence[mask_negative])
            if mask_negative.any()
            else torch.tensor([], device=data.device)
        )
        loss_uncertain = (
            self._uncertain_loss(data[mask_uncertain], confidence[mask_uncertain])
            if mask_uncertain.any()
            else torch.tensor([], device=data.device)
        )

        total_loss = torch.cat([loss_pos, loss_neg, loss_uncertain], dim=0)

        if total_loss.numel() == 0:
            return torch.tensor(0.0, device=data.device)

        return total_loss.mean()

    def score(self, batch: torch.Tensor) -> torch.Tensor:
        x_hat, _, kl_div = self.model(batch)
        return self.criterion(x_hat, kl_div, batch)

    def _positive_loss(self, data, confidence):
        scores = self.score(data)
        return scores

    def _negative_loss(self, data, confidence):
        scores = self.score(data)
        return abs(self.m - scores)

    def _uncertain_loss(self, data, confidence):
        scores = self.score(data)
        sorted_scores = scores.sort().values
        # eCDF(xi) = (1/n) * sum_j 1_{xj <= xi}
        ecdf = torch.searchsorted(sorted_scores, scores, right=True).float() / len(
            scores
        )
        loss_uncertain = ecdf * abs(self.m - scores)
        return loss_uncertain

    def _prediction_score(self, data):
        return self.score(data)

    def update_thresholds(self) -> List[float]:
        """Return the decision threshold according to the selected strategy."""
        if self.threshold_strategy == "auto":
            config = getattr(self.confidence_estimator, "configuration", None)
            if config is not None:
                return [
                    float(config.abnormal.left),
                    float(config.normal.right),
                    (float(config.abnormal.left) + float(config.normal.right)) / 2,
                ]
            logger.warning(
                "'auto' strategy selected but confidence intervals are missing or invalid. "
                "Falling back to manual threshold."
            )
        return [self.manual_threshold]
