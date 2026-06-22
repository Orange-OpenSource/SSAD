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
"""Public API for self-supervision distribution analyzers."""
from .supports_distribution_analysis import SupportsDistributionAnalysis
from .triangular_thresholding import TriangularThresholding
from .evt_thresholding import EVTThresholding

__all__ = [
    "SupportsDistributionAnalysis",
    "TriangularThresholding",
    "EVTThresholding",
]
