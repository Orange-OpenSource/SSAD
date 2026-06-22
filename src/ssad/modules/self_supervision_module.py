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
Self Supervision Module
"""

import logging
import time
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Sequence, Union

import lightning as L
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,  # type: ignore[import-untyped]
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import Tensor, nn
from typing_extensions import override

from ssad.confidence_estimators.supports_confidence_estimation import (
    SupportsConfidenceEstimation,
)
from ssad.datamodules.dataset_with_confidence import DatasetWithConfidence
from ssad.datamodules.self_supervision_datamodule import SelfSupervisionDataModule
from ssad.distribution_analyzers.supports_distribution_analysis import (
    SupportsDistributionAnalysis,
)
from ssad.loggers.mlflow_logger import (
    get_mlflow_logger,
    log_confidence,
    log_confidence_analysis,
    log_confidence_intervals,
    log_system_metrics,
    log_test_metrics,
)

from .supports_self_supervision import SupportsSelfSupervision

## use of prototype masked tensor API, may require maintenance down the road
# from torch.masked import MaskedTensor, masked_tensor

# Disable prototype warnings and such
warnings.filterwarnings(action="ignore", category=UserWarning)
logger = logging.getLogger(__name__)


class SelfSupervisionCallback(L.Callback):
    """
    Lightning callback used to regularly update confidence in training samples during training.
    """

    def on_fit_start(self, trainer, pl_module):  # pylint: disable=unused-argument
        client = mlflow.tracking.MlflowClient()
        run_id = trainer.logger.run_id
        tags_to_log = {
            "Batch Size ": trainer.datamodule.datamodule.batch_size,
            "Dataset Size ": trainer.datamodule.datamodule.n_rows,
            "Train Percentage ": trainer.datamodule.datamodule.p_train,
            "Val Percentage ": trainer.datamodule.datamodule.p_val,
            "Anomaly Train Percentage ": trainer.datamodule.datamodule.pa_train,
            "Anomaly Val Percentage ": trainer.datamodule.datamodule.pa_val,
            "Anomaly Test Percentage ": trainer.datamodule.datamodule.pa_test,
        }
        for key, value in tags_to_log.items():
            client.set_tag(run_id, key, str(value))

    @override
    def on_train_epoch_start(
        self, trainer, pl_module
    ):  # pylint: disable=unused-argument
        pl_module.epoch_start_time = time.time()

    @override
    def on_train_epoch_end(self, trainer, pl_module):

        if not isinstance(pl_module, SelfSupervisionModule):
            raise TypeError(
                "Wrong type for module, expected SelfSupervisionModule, "
                f"got {pl_module.__class__.__name__}"
            )

        if not isinstance(trainer.datamodule, SelfSupervisionDataModule):
            raise TypeError(
                "Wrong type for trainer datamodule, expected SelfSupervisionDataModule, "
                f"got {trainer.datamodule.__class__.__name__}"
            )

        # Epoch Duration
        duration = time.time() - getattr(pl_module, "epoch_start_time", time.time())
        logger.info(
            "⏱️ Epoch %d duration: %.2f seconds", trainer.current_epoch, duration
        )

        if trainer.logger is not None:
            mlf_logger = get_mlflow_logger(trainer)
            if mlf_logger is None:
                pass
            else:
                mlf_logger.log_metrics(
                    {"epoch_duration_seconds": duration}, step=trainer.current_epoch
                )
                log_system_metrics(mlf_logger, epoch=trainer.current_epoch)

        if (
            trainer.current_epoch % pl_module.every_n_epochs == 0
            and trainer.current_epoch != 0
        ):
            logger.info("🔄 Updating Confidence scores and Intervals.")
            with torch.no_grad():
                confidence = pl_module.update_confidence(trainer.datamodule)
                trainer.datamodule.train = DatasetWithConfidence(
                    trainer.datamodule.train.dataset, confidence
                )

            if trainer.logger is not None:
                mlf_logger = get_mlflow_logger(trainer)
                if mlf_logger is None:
                    return

                log_confidence_analysis(
                    pl_module.distr_analyzer, mlf_logger, trainer.current_epoch
                )
                log_confidence_intervals(
                    pl_module.confidence_estimator, mlf_logger, trainer.current_epoch
                )
                log_confidence(
                    mlf_logger, trainer.datamodule.train, trainer.current_epoch
                )


class SelfSupervisionModule(
    L.LightningModule, ABC, SupportsSelfSupervision
):  # pylint: disable=too-many-instance-attributes
    """Lightning module implementing self-supervision.

    Self-supervision consists in leveraging a representation of an input extracted
    from a model to estimate the confidence in the fact that the input is normal or
    abnormal.

    This information is used during training by applying different loss functions
    depending on the confidence estimation.:
    - _positive_loss: for normal samples
    - _negative_loss: for abnormal samples
    These functions may also take into account the confidence estimation to weight
    the final loss.
    """

    # @property
    # @abstractmethod
    # def criterion(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    #     """Criterion to determine the model score by comparing a sample and its
    #     model output.
    #     This criterion should be a function with the following signature:
    #     def 'func'(torch.Tensor, torch.Tensor) -> torch.tensor.

    #     Returns:
    #         Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: criterion function
    #     """
    @property
    @abstractmethod
    def criterion(self) -> Callable[..., torch.Tensor]:
        """Criterion to compute a score between model outputs and inputs."""
        raise NotImplementedError()

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        model: nn.Module,
        every_n_epochs: int,
        confidence_estimator: SupportsConfidenceEstimation,
        distr_analyzer: SupportsDistributionAnalysis,
        threshold_strategy: str = "auto",  # "manual" or "auto"
        manual_threshold: float = 0.5,
    ):
        """
        Initializes the supervision module based on confidence estimation and distribution analysis.

        Args:
            model (nn.Module): The PyTorch model used to generate representations or predictions.
            every_n_epochs (int):
                Number of epochs between each confidence and distribution analysis.
            confidence_estimator (SupportsConfidenceEstimation):
                Object responsible for estimating confidence scores from the model's outputs.
            distr_analyzer (SupportsDistributionAnalysis):
                Object responsible for analyzing the distribution of confidence scores.
            threshold_strategy (str): Strategy for determining the anomaly detection threshold
                for val/test steps. Should be either "manual" or "auto".
            manual_threshold (float):
                Threshold value used when `threshold_strategy` is set to "manual".

        Attributes
        ----------
            val_outputs (Dict[str, List[Tensor]]):
                Dictionary storing validation scores and corresponding labels.
            test_outputs (Dict[str, List[Tensor]]):
                Dictionary storing test scores and corresponding labels.
        """
        super().__init__()
        self.model = model
        self.every_n_epochs = every_n_epochs
        self.confidence_estimator = confidence_estimator
        self.distr_analyzer = distr_analyzer
        if threshold_strategy not in {"manual", "auto"}:
            raise ValueError("threshold_strategy must be 'manual' or 'auto'")
        self.threshold_strategy = threshold_strategy
        self.manual_threshold = manual_threshold
        self.val_outputs: Dict[str, List[Tensor]] = {"scores": [], "labels": []}
        self.test_outputs: Dict[str, List[Tensor]] = {"scores": [], "labels": []}
        self.thresholds: List[float] = [manual_threshold]

    @abstractmethod
    def _positive_loss(
        self, data: torch.Tensor, confidence: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def _negative_loss(
        self, data: torch.Tensor, confidence: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def _uncertain_loss(
        self, data: torch.Tensor, confidence: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def _prediction_score(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    # TODO: generic compute_loss implementation
    @abstractmethod
    def compute_loss(
        self, data: torch.Tensor, confidence: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        """Compute the loss for a batch, given data and confidence."""
        raise NotImplementedError()

    @abstractmethod
    def update_thresholds(self) -> List[float]:
        """Return the decision threshold according to the selected strategy."""
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def forward(self, i, *args, **kwargs):  # pylint: disable=unused-argument
        return self.model.forward(i)

    def training_step(
        self, batch, batch_idx, *args, **kwargs
    ):  # pylint: disable=unused-argument

        # Squeeze removes dimensions of size 1 (e.g., from [32, 1, 28, 28] to [32, 28, 28]);
        # Such extra dimensions often come from how the dataset was originally created or loaded.
        data = batch["data"].squeeze()
        if torch.isnan(data).any():
            raise ValueError("Batch contains NaNs")

        confidence = batch["confidence"]

        while confidence.dim() < data.dim():
            confidence = confidence.unsqueeze(-1)

        loss = self.compute_loss(data, confidence, batch_idx)

        if batch_idx == 0:
            logger.debug("confidence shape : %s", confidence.shape)
            logger.debug("data shape : %s", data.shape)
            logger.debug("loss shape : %s -- loss : %s", loss.shape, loss)
            # for i in range(10):
            #     self.log(f"confidence_{i}", confidence[i].item(), on_step=False, on_epoch=True)

        self.log("train_loss", loss.item(), on_epoch=True, on_step=True, logger=True)

        return loss

    @abstractmethod
    def score(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
        # return self.criterion(self.model.forward(batch), batch)

    def configure_callbacks(self) -> Union[Sequence[L.Callback], L.Callback]:
        trainer_callbacks = getattr(self.trainer, "callbacks", []) if self.trainer else []
        already_present = any(isinstance(cb, SelfSupervisionCallback) for cb in trainer_callbacks)

        if already_present:
            return []

        return [SelfSupervisionCallback()]

    @torch.no_grad()
    def update_confidence(self, datamodule: SelfSupervisionDataModule) -> torch.Tensor:
        """Retrieves the distribution of confidence scores for the training dataset.
        Computes normal/abnormal/unknown intervals from the analysis of this distribution.
        Reconfigures the confidence estimator with these intervals.

        Args:
            datamodule (SelfSupervisionDataModule):
                A data module providing access to the supervision dataloader.

        Returns:
            confidence (torch.Tensor):
                A tensor containing the estimated confidence scores for all supervision samples.
        """
        self.model.eval()

        # FIXME: prevent complete score loading in memory, no guarantee to work
        # TODO: is self.distribution really necessary?
        # use the data saved on disk for distribution analysis
        scores = []
        supervision_dataloader = datamodule.supervision_dataloader()
        for batch in supervision_dataloader:
            batch = self.transfer_batch_to_device(batch, self.device, 0)
            batch_data = batch["data"].squeeze(1)
            batch_scores = self.score(batch_data)
            scores.append(batch_scores)

        # scores_batch goes from gpu to cpu here
        scores_batch = torch.cat(scores)
        configuration = self.distr_analyzer.analyze_distribution(
            scores_batch=scores_batch,
            current_conf=self.confidence_estimator.configuration,
        )
        self.confidence_estimator.configuration = configuration

        self.thresholds = self.update_thresholds()

        confidence = self.confidence_estimator.estimate_confidence(scores_batch)
        return confidence

    def validation_step(
        self, batch, _batch_idx, *args, **kwargs
    ):  # pylint: disable=unused-argument
        if isinstance(batch, (tuple, list)):
            x, y = batch
        elif isinstance(batch, dict):
            x = batch["data"]
            y = batch["label"]
        else:
            raise TypeError(f"Batch type unsupported: {type(batch)}")

        x = x.squeeze()
        y = y.view(-1)

        score = self._prediction_score(x)

        # We store the results as attributes
        self.val_outputs["scores"].append(score.cpu())
        self.val_outputs["labels"].append(y.cpu())

    def on_validation_epoch_end(self):
        if self.current_epoch == 0:
            return

        scores = torch.cat(self.val_outputs["scores"]).numpy()
        labels = torch.cat(self.val_outputs["labels"]).numpy()

        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)

        threshold = self.thresholds[0]

        # Compute predictions
        preds = (scores > threshold).astype(int)

        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        metrics = {
            "val_auc": auc,
            "val_ap": ap,
            "val_accuracy": acc,
            "val_precision": prec,
            "val_recall": rec,
            "val_f1": f1,
        }

        mlf_logger = get_mlflow_logger(self.trainer)
        if mlf_logger is not None:
            mlf_logger.log_metrics(metrics, step=self.current_epoch)

        # Freeing Memory
        self.val_outputs = {"scores": [], "labels": []}

    def test_step(
        self, batch, _batch_idx, *args, **kwargs
    ):  # pylint: disable=unused-argument

        if isinstance(batch, (tuple, list)):
            x, y = batch
        elif isinstance(batch, dict):
            x = batch["data"]
            y = batch["label"]
        else:
            raise TypeError("Unexpected batch format")

        score = self._prediction_score(x)

        # We store the results as attributes
        self.test_outputs["scores"].append(score.cpu())
        self.test_outputs["labels"].append(y.cpu())

    def on_test_epoch_end(self):

        scores = torch.cat(self.test_outputs["scores"]).numpy()
        labels = torch.cat(self.test_outputs["labels"]).numpy()

        mlf_logger = get_mlflow_logger(self.trainer)

        for threshold in self.thresholds:
            metrics = self.compute_test_metrics(scores, labels, threshold)
            if mlf_logger is not None:
                log_test_metrics(metrics, mlf_logger, threshold)

        self.plot_test_metrics(scores, labels, mlf_logger)

        # Freeing Memory
        self.test_outputs = {"scores": [], "labels": []}

    def compute_test_metrics(self, scores, labels, threshold, eps=1e-10):
        # Compute predictions
        preds = (scores > threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        fpr = fp / (fp + tn + eps)
        specificity = tn / (tn + fp + eps)
        fnr = fn / (fn + tp + eps)

        # Results as a dictionary
        metrics = {
            "threshold": threshold,
            "True Positives (TP)": tp,
            "True Negatives (TN)": tn,
            "False Positives (FP)": fp,
            "False Negatives (FN)": fn,
            "Accuracy": acc,
            "Precision": prec,
            "F1 Score": f1,
            "Recall (TPR / Sensitivity)": rec,
            "Specificity (TNR)": specificity,
            "False Positive Rate (FPR)": fpr,
            "False Negative Rate (FNR)": fnr,
        }

        return metrics

    def plot_test_metrics(self, scores, labels, mlf_logger):
        # === ROC & PR ===
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)

        fpr, tpr, thresholds_roc = roc_curve(labels, scores)

        roc_indices = np.linspace(0, len(thresholds_roc) - 1, 5, dtype=int)
        roc_points = [(fpr[i], tpr[i], thresholds_roc[i]) for i in roc_indices]

        fig_roc = plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], "k--", label="Random")

        for x, y, t in roc_points:
            plt.plot(x, y, "o")
            plt.text(x, y, f"{t:.2f}", fontsize=8)

        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC")
        plt.legend()
        plt.grid(True)

        # Direct Log via object figure
        if mlf_logger is not None:
            mlf_logger.experiment.log_figure(
                mlf_logger.run_id, fig_roc, artifact_file="test_roc.png"
            )
        plt.close(fig_roc)

        precs, recalls, thresholds_pr = precision_recall_curve(labels, scores)

        pr_indices = np.linspace(0, len(thresholds_pr) - 1, 5, dtype=int)
        pr_points = [(recalls[i], precs[i], thresholds_pr[i]) for i in pr_indices]

        fig_pr = plt.figure()
        plt.plot(recalls, precs, label=f"AP = {ap:.2f}")

        for x, y, t in pr_points:
            plt.plot(x, y, "o")
            plt.text(x, y, f"{t:.2f}", fontsize=8)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PRC")
        plt.legend()
        plt.grid(True)
        if mlf_logger is not None:
            mlf_logger.experiment.log_figure(
                mlf_logger.run_id, fig_pr, artifact_file="test_pr.png"
            )
        plt.close(fig_pr)
