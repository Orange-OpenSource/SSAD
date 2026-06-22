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
Initializes the main package API by exposing public interfaces from submodules.

This module re-exports the core components of the package to simplify access and
maintain a clean and consistent public API.

Available namespaces:
- confidence_estimators
- datamodules
- datasets
- distribution_analyzers
- models
- modules
- loggers

Usage:
    from mypackage import SomeModel, SomeDataModule, ConfidenceEstimator

Note:
    This file uses wildcard imports (`*`) to expose only the public symbols defined
    in each submodule's `__all__` list.
"""
from .confidence_estimators.api import *
from .datamodules.api import *
from .datasets.api import *
from .distribution_analyzers.api import *
from .models.api import *
from .modules.api import *
from .loggers.api import *
