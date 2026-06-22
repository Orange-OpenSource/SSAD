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
Logging configuration module.

This module provides a simple setup function to configure the logging system
with a console handler and a customizable log level.

The default formatter displays logs in the format: [LEVEL]: message

Example:
    from logging_config import setup_logging
    setup_logging(level='DEBUG')
"""

import logging.config


def setup_logging(level="INFO"):
    """
    Configures the root logger with a console output and the specified log level.

    Args:
        level (str): Logging level to apply (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR').

    The logger uses a simple format: [LEVEL]: message.
    Existing loggers are preserved and not disabled.
    """
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "[%(levelname)s]: %(message)s in %(name)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": level,
            },
        },
        "root": {
            "handlers": ["console"],
            "level": level,
        },
    }
    logging.config.dictConfig(config)
