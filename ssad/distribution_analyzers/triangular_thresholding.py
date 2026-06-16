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
Implements a triangular thresholding supervisor.
Triangular thresholding is the implementation of a unimodal thresholding method proposed by
Rosin, Paul. (2001). Unimodal thresholding. Pattern Recognition. 34. 2083-2096.
10.1016/S0031-3203(00)00136-9.
"""

from __future__ import annotations

from math import lgamma
from typing import Any, Optional, TYPE_CHECKING
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch

from ssad.confidence_estimators.confidence_intervals_configuration import (
    ConfidenceIntervalsConfiguration,
    Interval,
)

from .supports_distribution_analysis import SupportsDistributionAnalysis

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TriangularThresholding(SupportsDistributionAnalysis):
    """Supervisor using the Triangular thresholding heuristic.

    Supposes that the distribution values are between 0 and 1."""

    def __init__(
        self,
        bin_estimator: str = "knuth",
        abnormal_threshold: Optional[float] = None,
        normal_threshold: Optional[float] = None,
    ):
        self.bin_estimator = bin_estimator
        self.counts: Optional[NDArray[np.int_]] = None
        self.bin_edges: Optional[NDArray[np.floating[Any]]] = None
        self.abnormal_threshold = abnormal_threshold
        self.normal_threshold = normal_threshold

    @torch.no_grad()
    def analyze_distribution(self, scores_batch, current_conf):

        scores_batch_array = scores_batch.cpu().numpy()

        if not np.all((scores_batch_array >= 0) & (scores_batch_array <= 1)):
            raise ValueError("Wrong value in distribution, should be between 0 and 1.")

        self.counts, self.bin_edges = self.get_histogram(scores_batch_array)

        if current_conf is None:
            # initialize estimator with a single threshold
            _idx_normal, self.normal_threshold = unimodal_left(
                self.counts, self.bin_edges
            )
            self.abnormal_threshold = self.normal_threshold
        else:
            # compute thresholds for left and right halves of the distribution histogram
            # left is close to 0 and corresponds to abnormal scores
            # right is close to 1 and corresponds to normal scores
            cutting_index = len(self.counts) // 2
            _idx_normal, self.normal_threshold = unimodal_left(
                self.counts[cutting_index:], self.bin_edges[cutting_index:]
            )
            _idx_normal += cutting_index

            _idx_abnormal, self.abnormal_threshold = unimodal_right(
                self.counts[:cutting_index], self.bin_edges[: cutting_index + 1]
            )

        normal_interval = Interval(left=self.normal_threshold, right=1, closed="right")
        abnormal_interval = Interval(left=0, right=self.abnormal_threshold)
        unknown_interval = Interval(
            left=self.abnormal_threshold, right=self.normal_threshold, closed="both"
        )
        return ConfidenceIntervalsConfiguration(
            normal=normal_interval, abnormal=abnormal_interval, unknown=unknown_interval
        )

    def plot_analysis(
        self, style="default", plot_size=(16, 22)
    ) -> matplotlib.figure.Figure:
        """plots a figure visualizing the supervising distribution.
        Corresponds to a histogram of reconstruction scores, with thresholds
        for normal and abnormal intervals.

        Args:
            style (str, optional): matplotlib style for plot. Defaults to "default".
            plot_size (tuple, optional): size of plot. Defaults to (16,22).

        Returns:
            matplotlib.figure.Figure: Figure with histogram and thresholds.
        """
        with plt.style.context(style=style):
            fig, ax = plt.subplots(figsize=plot_size)
            if self.counts is not None and self.bin_edges is not None:
                ax.stairs(self.counts, self.bin_edges)
            if self.normal_threshold is not None:
                ax.axvline(self.normal_threshold, color="blue")
            if self.abnormal_threshold is not None:
                ax.axvline(self.abnormal_threshold, color="red")

            # Set labels and title
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

    def get_histogram(
        self, distribution_array: NDArray[Any]
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        r"""
        returns the histogram according to the number of bins obtained by the bin number estimator
        two values are returned:
            counts: array of counts for each bin
            bin_edges: array of bin edges, of length equal to len(counts)+1

        accepted bin estimators:
        - knuth: implemented in knuth_bin_histogram and knuth_criterion
        - auto : maximum of the 'sturges' and 'fd' estimators
        - fd : Freedman Diaconis Estimator, h = 2 \frac{IQR}{n^{1/3}}
        - scott : h = \sigma \sqrt[3]{\frac{24 \sqrt{\pi}}{n}}
        - rice : n_h = 2n^{1/3}
        - sturges : n_h = \log _{2}(n) + 1
        - doane : n_h = 1 + \log_{2}(n) +
                        \log_{2}\left(1 + \frac{|g_1|}{\sigma_{g_1}}\right)
                g_1 = mean\left[\left(\frac{x - \mu}{\sigma}\right)^3\right]
                \sigma_{g_1} = \sqrt{\frac{6(n - 2)}{(n + 1)(n + 3)}}
        - sqrt :  n_h = \sqrt n
        """
        if self.bin_estimator == "knuth":
            return knuth_bin_histogram(
                data=distribution_array, min_bins=10, max_bins=250, step_search=1
            )

        return np.histogram(a=distribution_array, bins=self.bin_estimator)


def unimodal_left(counts, bin_edges, reverse: bool = False) -> tuple[int, float]:
    """Implements the unimodal histogram thresholding strategy
    Thresholding is performed to the left of the peak.
    Use reverse = True to perform a unimodal thresholding to the right.

    Args:
        counts (List[int]): counts for each bin of the histogram
        bin_edges (List[float]): edges of the bins
        reverse (bool, optional): defaults to False. Set True to threshold to the right.

    Returns:
        Tuple[int, float]: index of threshold bin, threshold value
    """

    # if threshold to the right, reverse the histogram
    if reverse:
        counts = counts[::-1]
        bin_edges = bin_edges[::-1]

    # compute bin centers from start_index to end_index
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # find peak bin: index, value, and coordinates of center
    index_peak = np.argmax(counts)
    count_peak = counts[index_peak]
    peak_coordinates = np.array([bin_centers[index_peak], count_peak])

    # find the index of the last bin with zero count
    index_last_zero = 0
    for index, count in enumerate(counts):
        if count == 0:
            index_last_zero = index
        else:
            break

    # coordinates of the center of the last bin with zero count
    last_zero_coordinates = np.array([bin_centers[index_last_zero], 0])

    # a: vector from last zero bin center to peak bin center
    a = peak_coordinates - last_zero_coordinates
    norm_a = np.linalg.norm(a)

    # find index of bin whose center is at the max distance to the line a
    # distance of point P to line with points AB: abs(cross_product(AP, AB)) / norm(AB)
    index_threshold_bin = -1
    max_distance = -1
    for index in range(index_last_zero, index_peak):
        candidate_peak_coordinates = np.array([bin_centers[index], counts[index]])
        # b: vector from last zero bin center to candidate peak
        b = candidate_peak_coordinates - last_zero_coordinates
        cross_ab = a[0] * b[1] - b[0] * a[1]
        distance = abs(cross_ab) / norm_a
        if distance > max_distance:
            index_threshold_bin = index
            max_distance = distance

    threshold_bin = bin_centers[index_threshold_bin]
    # if histogram was reversed, compute the correct index for the threshold bin
    if reverse:
        index_threshold_bin = len(counts) - 1 - index_threshold_bin

    # return index_threshold_bin, bin_centers[index_threshold_bin]
    return index_threshold_bin, threshold_bin


def unimodal_right(counts, bin_edges):
    """Implements the unimodal histogram thresholding strategy
    Thresholding is performed to the right of the peak.
    Shortcut for unimodal_left(counts, bin_edges, reverse=True)

    Args:
        counts (List[int]): counts for each bin of the histogram
        bin_edges (List[float]): edges of the bins

    Returns:
        float: threshold value
    """
    return unimodal_left(counts, bin_edges, reverse=True)


# Knuth bin estimator
def knuth_bin_histogram(
    data: NDArray[Any], min_bins: int, max_bins: int, step_search: int
) -> tuple[NDArray[Any], NDArray[Any]]:
    """_summary_

    Args:
        data (_type_): _description_
        min_bins (_type_): _description_
        max_bins (_type_): _description_
        step_search (_type_): _description_

    Returns:
        _type_: _description_
    """

    if data.ndim != 1:
        raise ValueError("Knuth criterion requires 1D input")

    min_knuth_cost = np.inf
    optimal_counts = None
    optimal_bins = None

    # perform grid search to find optimal number of bins to minimize the Knuth criterion
    for num_bins in range(min_bins, max_bins + 1, step_search):
        counts, bins = np.histogram(data, bins=num_bins)
        knuth_cost = knuth_criterion(data.size, num_bins, num_bins)
        if knuth_cost < min_knuth_cost:
            min_knuth_cost = knuth_cost
            optimal_counts = counts
            optimal_bins = bins

    if optimal_bins is None:
        raise ValueError("Optimal bins should not be None")

    if optimal_counts is None:
        raise ValueError("Optimal counts should not be None")

    return optimal_counts, optimal_bins


def knuth_criterion(data_size: int, num_bins: int, counts: int) -> float:
    """
    implementation of K. H. Knuth, 'Optimal data-based binning for histograms
    and histogram-based probability density models', Digital Signal Processing,
    vol. 95, p. 102581, déc. 2019, doi: 10.1016/j.dsp.2019.102581.
    """
    return -1 * (
        data_size * np.log(num_bins)
        + lgamma(0.5 * num_bins)
        - num_bins * lgamma(0.5)
        - lgamma(data_size + 0.5 * num_bins)
        + np.sum(lgamma(counts + 0.5))
    )
