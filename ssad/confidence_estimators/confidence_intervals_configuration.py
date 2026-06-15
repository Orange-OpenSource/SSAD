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
A Confidence Intervals configuration consists in a set of three intervals
for the value of the analyzed distribution (e.g., reconstruction scores).
These three intervals respectively define the intervals that correspond to
normal samples, abnormal samples, and samples for which it is impossible to decide,
labelled "unknown".
"""

from dataclasses import dataclass

import pandas as pd
import torch


class Interval(pd.Interval):
    """Extends pandas Intervals to add a method to get tensor masks corresponding to values
    within the interval.
    """

    def contains_tensor_mask(self, item: torch.Tensor) -> torch.Tensor:
        """Computes the mask corresponding to the tensor indexes with a value contained within the interval.

        Args:
            item (torch.Tensor): tensor whose mask is to be computed

        Returns:
            torch.Tensor: mask, boolean tensor with the same shape as item,
                where values are True when the corresponding value (same index) in item
                is within the Interval.
        """
        left_tensor = item >= self.left if self.closed_left else item > self.left
        right_tensor = item <= self.right if self.closed_right else item > self.right
        return torch.logical_and(left_tensor, right_tensor)


@dataclass
class ConfidenceIntervalsConfiguration:
    """Represents the configuration of intervals for a confidence estimator."""

    normal: Interval
    abnormal: Interval
    unknown: Interval

    def as_dict(self) -> dict[str, str]:
        """Provides a dict representation of the intervals configuration

        Returns:
            dict[IntervalLiteral, str]:
                dictionary of string representations for intervals in the configuration.
        """
        dict_repr = {
            "normal": str(self.normal),
            "abnormal": str(self.abnormal),
            "unknown": str(self.unknown),
        }
        return dict_repr
