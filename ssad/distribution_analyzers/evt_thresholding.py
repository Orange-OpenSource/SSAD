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
EVT-based thresholding supervisor using the Peaks-Over-Threshold (POT) approach.

This module implements a supervisor strategy for confidence score calibration based on
Extreme Value Theory (EVT). It identifies extreme values in the score distribution by
modeling the tail using the Generalized Pareto Distribution (GPD) and separating 
samples into three categories: nominal, unknown, and abnormal.
"""
import logging
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import genpareto # type: ignore[import-untyped]

from ssad.confidence_estimators.confidence_intervals_configuration import (
    ConfidenceIntervalsConfiguration,
    Interval,
)

from .supports_distribution_analysis import SupportsDistributionAnalysis

logger = logging.getLogger(__name__)

class EVTThresholding(SupportsDistributionAnalysis):
    """
    Supervisor using the Peaks-Over-Threshold (POT) method from Extreme Value Theory (EVT).

    This strategy models the tail of the score distribution with a Generalized Pareto 
    Distribution (GPD) to detect extreme values (potential anomalies). It sets an initial 
    threshold u and a high threshold tq to define three regions in the data:
    - Normal: scores <= u
    - Unknown: u < scores <= tq
    - Abnormal: scores > tq
    """
    def __init__(self, alpha: float = 1.5, q: float = 0.001):
        """
        Initialize the EVTThresholding supervisor.

        Args:
            alpha (float): Scale factor for the interquartile range to compute threshold u.
            q (float): Risk level used to compute the high threshold tq.
        """
        self.alpha = alpha
        self.q = q
        self.u = None
        self.tq = None
        self.hist_counts = None
        self.hist_edges = None


    @torch.no_grad()
    def analyze_distribution(self, scores_batch: torch.Tensor, current_conf):
        """
        Analyze the score distribution and return confidence intervals.

        Applies the POT approach:
        - Estimate u = Q3 + alpha * IQR
        - Fit a Generalized Pareto Distribution (GPD) on values > u
        - Estimate tq using the fitted GPD and risk parameter q

        Args:
            scores_batch (torch.Tensor): A 1D tensor of scores (values ∈ [0, 1])
            current_conf (Optional[ConfidenceIntervalsConfiguration]): 
                Unused, kept for compatibility

        Returns:
            ConfidenceIntervalsConfiguration: 
                Object defining normal, abnormal, and unknown intervals
        """
        scores_np = scores_batch.cpu().numpy()

        Q1 = np.percentile(scores_np, 25)       # pylint: disable=invalid-name
        Q3 = np.percentile(scores_np, 75)       # pylint: disable=invalid-name
        IQR = Q3 - Q1                           # pylint: disable=invalid-name
        self.u = Q3 + self.alpha * IQR
        logger.info("limit u : %s", self.u)

        exceedances = scores_np[scores_np > self.u] - self.u
        N = len(exceedances)                    # pylint: disable=invalid-name
        n = len(scores_np)

        if N == 0:
            raise ValueError("No values exceed threshold u. Try lowering alpha.")

        shape, _, scale = genpareto.fit(exceedances, floc=0)

        if shape != 0:
            self.tq = self.u + (scale / shape) * (((self.q * N) / n) ** (-shape) - 1)
        else:
            self.tq = self.u + scale * np.log(n / (self.q * N))

        logger.info("limit tq : %s", self.tq)

        # Save histogram for plotting
        self.hist_counts, self.hist_edges = np.histogram(scores_np, bins="auto")

        # Define the intervals
        normal = Interval(left=0, right=self.u)
        abnormal = Interval(left=self.tq, right=np.inf, closed="left")
        unknown = Interval(left=self.u, right=self.tq, closed="both")

        return ConfidenceIntervalsConfiguration(
            normal=normal, abnormal=abnormal, unknown=unknown
        )

    def plot_analysis(self, style="default", plot_size=(16, 8)) -> matplotlib.figure.Figure:
        """
        Plot histogram of the score distribution with u and tq thresholds.

        Args:
            style (str, optional): Matplotlib style. Defaults to "default".
            plot_size (tuple, optional): Size of the figure. Defaults to (16, 8).

        Returns:
            matplotlib.figure.Figure: The generated figure object.
        """
        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=plot_size)

            if self.hist_counts is not None and self.hist_edges is not None:
                ax.stairs(self.hist_counts, self.hist_edges, label="Histogram")
            if self.u is not None:
                ax.axvline(self.u, color="orange", linestyle="--", label="Threshold u")
            if self.tq is not None:
                ax.axvline(self.tq, color="red", linestyle="--", label="Threshold tq")

            ax.set_title(
                "Histogram of confidence scores with abnormal (red) and normal (blue) thresholds",
                fontsize=14,
            )
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            ax.legend()
            plt.tight_layout()

        plt.close(fig)
        return fig
