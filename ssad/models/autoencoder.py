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
Implementation of a fully connected feed-forward autoencoder
"""

# pylint: disable=duplicate-code
# Init logic is intentionally similar across autoencoders to keep implementations independent.
from typing import List, Optional
from typing_extensions import override
import torch


class Autoencoder(torch.nn.Module):
    """
    torch.nn.Module for autoencoders
    """

    # TODO add decoder_dims which is the reverse of encoder_dims if None
    def __init__(
        self, input_dim: int, encoder_dims: Optional[List[int]], latent_dim: int
    ):
        super().__init__()

        if encoder_dims is None:
            encoder_dims = []

        if latent_dim <= 0:
            raise ValueError(f"Wrong embedding dimension: {latent_dim}")

        if input_dim <= 0:
            raise ValueError(f"Wrong input dimension: {input_dim}")

        self.input_dim = input_dim
        self.encoder_dims = encoder_dims
        self.latent_dim = latent_dim

        dims = encoder_dims
        dims.insert(0, self.input_dim)
        dims.append(self.latent_dim)

        # encoder
        self.encoder = torch.nn.Sequential()
        for idx, layer_dim in enumerate(dims[:-1]):
            self.encoder.add_module(
                f"encoder_{idx}_linear", torch.nn.Linear(layer_dim, dims[idx + 1])
            )
            self.encoder.add_module(f"encoder_{idx}_relu", torch.nn.ReLU(inplace=False))

        # decoder
        reverse_dims = list(reversed(dims))
        reverse_dims_len = len(reverse_dims)
        self.decoder = torch.nn.Sequential()
        for idx, layer_dim in enumerate(reverse_dims[:-1]):
            self.decoder.add_module(
                f"decoder_linear_{idx}",
                torch.nn.Linear(layer_dim, reverse_dims[idx + 1]),
            )
            if idx != (reverse_dims_len - 2):
                self.decoder.add_module(
                    f"decoder_relu_{idx}", torch.nn.ReLU(inplace=False)
                )

        self.decoder.add_module("decoder_sigmoid", torch.nn.Sigmoid())

    @override
    def forward(self, forward_input: torch.Tensor):
        """computes the autoencoder output for a specific input

        Args:
            forward_input (torch.Tensor): input torch.Tensor for neural network

        Returns:
            torch.Tensor: output torch.Tensor
        """
        z = self.encoder(forward_input)
        output = self.decoder(z)
        return output
