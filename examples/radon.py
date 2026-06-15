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

N. Najari, S. Berlemont, G. Lefebvre, S. Duffner, et C. Garcia, « RADON: Robust Autoencoder for Unsupervised Anomaly Detection », in 2021 14th International Conference on Security of Information and Networks (SIN), déc. 2021, p. 1-8. doi: 10.1109/SIN54109.2021.9699174.

"""
import os
from datetime import datetime
import logging

import pytz
import torch
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger

import ssad
from datasets.mnist import MNISTAnomalyDataModule


batch_size = 256
n_rows=9000
p_train=0.4
p_val=0.1
pa_train=0.1
pa_val=0.5
pa_test=0.5

LOG_LEVEL = "WARNING" # INFO -- WARNING -- ERROR
DATASET_NAME = "MNIST"
MODEL_NAME = "RADON"

ssad.setup_logging(level=LOG_LEVEL)
logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

def format_run_name(dt: datetime):
    tz = dt.strftime("%z")
    tz_colon = f"{tz[:3]}:{tz[3:]}" if tz else ""
    return DATASET_NAME + " " + dt.strftime(f"%Y-%m-%d at %H:%M:%S ({tz_colon})")

utc_datetime = datetime.now(pytz.timezone("UTC"))
run_name = format_run_name(utc_datetime)
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

torch.set_float32_matmul_precision("high")

data = ssad.SelfSupervisionDataModule(
    datamodule=MNISTAnomalyDataModule(
        dataset_path=f"{ROOT_DIR}/data/MNIST/MNIST_original",
        preprocessed_dir=f"{ROOT_DIR}/data/MNIST/MNIST_processed",
        label_column="label",
        batch_size=batch_size,
        n_rows=n_rows,
        p_train=p_train,
        p_val=p_val,
        pa_train=pa_train,
        pa_val=pa_val,
        pa_test=pa_test,
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

L.seed_everything(42, workers = True)

trainer = L.Trainer(
    max_epochs=10,
    accelerator="auto",
    devices=1,
    precision=16,
    logger=MLFlowLogger(
        experiment_name=MODEL_NAME,
        run_name=run_name,
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
