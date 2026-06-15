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
Utility functions for logging confidence data, system metrics, and analysis outputs to MLflow.
"""
from typing import Optional, cast
import tempfile
from pathlib import Path
import torch
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.tracking import MlflowClient  # type: ignore[import-untyped]
import psutil


from ssad.confidence_estimators.supports_confidence_estimation import SupportsConfidenceEstimation
from ssad.datamodules.dataset_with_confidence import DatasetWithConfidence, save_confidence_to_csv
from ssad.distribution_analyzers.supports_distribution_analysis import SupportsDistributionAnalysis

def get_mlflow_logger(trainer: L.Trainer) -> Optional[MLFlowLogger]:
    """Safely get MLFlow logger from Trainer."""

    if isinstance(trainer.logger, MLFlowLogger):
        return trainer.logger

    if isinstance(trainer.logger, list):
        for logger in trainer.logger:
            if isinstance(logger, MLFlowLogger):
                return logger

    return None


def log_confidence(
    mlf_logger: MLFlowLogger,
    dataset: DatasetWithConfidence,
    num_epoch: Optional[int] = None,
):
    """Logs the current confidence values as a CSV file.

    Args:
        mlf_logger (MLFlowLogger): MLFlow logger
        dataset (DatasetWithConfidence): dataset with confidence to save
        num_epoch (Optional[int], optional): num_epoch. Defaults to None.
    """
    filename = f"confidence_epoch_{num_epoch}.csv"

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir, filename)
        save_confidence_to_csv(dataset, path)
        cast(MlflowClient, mlf_logger.experiment).log_artifact(mlf_logger.run_id, path)


def log_confidence_analysis(
    distr_analyzer: SupportsDistributionAnalysis,
    mlf_logger: MLFlowLogger,
    num_epoch: int,
) -> None:
    """Logs confidence distribution analysis figure to MLflow."""

    if mlf_logger.run_id is None:
        raise ValueError("Invalid run id.")

    cast(MlflowClient, mlf_logger.experiment).log_figure(
        run_id=mlf_logger.run_id,
        figure=distr_analyzer.plot_analysis(),
        artifact_file=f"confidence_analysis_epoch_{num_epoch}.svg",
    )


def log_confidence_intervals(
    confidence_estimator: SupportsConfidenceEstimation,
    mlf_logger: MLFlowLogger,
    num_epoch: int,
) -> None:
    """_summary_

    Args:
        mlf_logger (MLFlowLogger): _description_
        num_epoch (int): _description_
    """
    if mlf_logger.run_id is None:
        raise ValueError("Invalid run id.")

    if confidence_estimator.configuration is not None:
        cast(MlflowClient, mlf_logger.experiment).log_dict(
            mlf_logger.run_id,
            confidence_estimator.configuration.as_dict(),
            f"confidence_intervals_epoch_{num_epoch}.json",
        )



def log_system_metrics(mlf_logger : MLFlowLogger, epoch):
    """Logs CPU, RAM, and GPU usage metrics to MLflow."""

    # CPU Usage
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent

    # GPU Usage (si CUDA disponible)
    if torch.cuda.is_available():
        gpu_id = 0  # Adapter si plusieurs GPUs
        gpu_usage = torch.cuda.utilization(gpu_id) if hasattr(torch.cuda, 'utilization') else 0
        gpu_mem_alloc = torch.cuda.memory_allocated(gpu_id) / 1024**3  # En GB
        gpu_mem_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3  # En GB
    else:
        gpu_usage, gpu_mem_alloc, gpu_mem_reserved = 0, 0, 0

    # Log dans MLflow si logger actif
    mlf_logger.log_metrics({
        "cpu_usage_percent": cpu_usage,
        "ram_usage_percent": ram_usage,
        "gpu_usage_percent": gpu_usage,
        "gpu_memory_allocated_gb": gpu_mem_alloc,
        "gpu_memory_reserved_gb": gpu_mem_reserved,
    }, step=epoch)



def log_test_metrics(
    metrics : dict,
    mlf_logger: MLFlowLogger,
    threshold: float,
) -> None:
    """Logs test metrics."""
    if mlf_logger.run_id is None:
        raise ValueError("Invalid run id.")

    cast(MlflowClient, mlf_logger.experiment).log_dict(
        mlf_logger.run_id,
        metrics,
        f"test_metrics_threshold={threshold}.json",
    )

