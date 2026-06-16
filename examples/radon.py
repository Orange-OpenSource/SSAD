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
Implementation of the RADON model using the SSAD library.

N. Najari, S. Berlemont, G. Lefebvre, S. Duffner, et C. Garcia, 
« RADON: Robust Autoencoder for Unsupervised Anomaly Detection », 
in 2021 14th International Conference on Security of Information and Networks (SIN), 
déc. 2021, p. 1-8. doi: 10.1109/SIN54109.2021.9699174.

"""

import logging
import os

import lightning as L
import torch
from datasets.mnist import MNISTAnomalyDataModule
from lightning.pytorch.loggers import MLFlowLogger
from utils import run_name

import ssad

DATASET_NAME = "MNIST"
MODEL_NAME = "RADON"
RUN_NAME = run_name(DATASET_NAME)
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
LOG_LEVEL = "WARNING"  # INFO -- WARNING -- ERROR
RANDOM_STATE = 42
ssad.setup_logging(level=LOG_LEVEL)
logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# cf. ssad.GeneralTabularDatamodule
#   p_train (float): Proportion of total data for training.
#   p_val (float): Proportion of total data for validation.
#   pa_train (float): Proportion of anomalies in training set.
#   pa_val (float): Proportion of anomalies in validation set.
#   pa_test (float): Proportion of anomalies in test set.
#   n_rows (Optional[int], optional): Total number of data points to include.
#        Defaults to dataset size.
#        Limits the dataset size by restricting the total number of rows to n_rows if specified.
BATCH_SIZE = 256
N_ROWS = 5000
P_TRAIN = 0.4
P_VAL = 0.1
PA_TRAIN = 0.1
PA_VAL = 0.5
PA_TEST = 0.5

torch.set_float32_matmul_precision("high")

data = ssad.SelfSupervisionDataModule(
    datamodule=MNISTAnomalyDataModule(
        dataset_path=f"{ROOT_DIR}/data/MNIST/MNIST_original",
        preprocessed_dir=f"{ROOT_DIR}/data/MNIST/MNIST_processed",
        label_column="label",
        batch_size=BATCH_SIZE,
        n_rows=N_ROWS,
        p_train=P_TRAIN,
        p_val=P_VAL,
        pa_train=PA_TRAIN,
        pa_val=PA_VAL,
        pa_test=PA_TEST,
        transform=ssad.DataFrameToTensor(),
        target_transform=ssad.DataFrameToTensor(),
        random_state=42,
    )
)

model = ssad.CosineReconstructionModule(
    model=ssad.Autoencoder(
        input_dim=data.input_dim,
        encoder_dims=[400],
        latent_dim=200,
    ),
    every_n_epochs=3,
    confidence_estimator=ssad.BinaryConfidence(),
    distr_analyzer=ssad.TriangularThresholding(bin_estimator="knuth"),
)

L.seed_everything(42, workers=True)

trainer = L.Trainer(
    max_epochs=10,
    accelerator="auto",
    devices=1,
    precision=16,
    logger=MLFlowLogger(
        experiment_name=MODEL_NAME,
        run_name=RUN_NAME,
        log_model=True,
        tracking_uri=f"file:{ROOT_DIR}/ml-runs",
    ),
    reload_dataloaders_every_n_epochs=1,
)

trainer.fit(
    model=model,
    datamodule=data,
)

trainer.test(model=model, datamodule=data)
