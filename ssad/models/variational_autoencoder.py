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
Implementation of a feed-forward, fully-connected variational autoencoder
"""

# pylint: disable=duplicate-code
# Init logic is intentionally similar across autoencoders to keep implementations independent.
import logging
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
LOG2PI = torch.log(torch.tensor(2.0 * torch.pi))


class VariationalAutoencoder(nn.Module):  # pylint: disable=too-many-instance-attributes
    """
    Variational Autoencoder with Normalizing Flows for enhanced posterior flexibility.

    Attributes:
        input_dim (int): Dimensionality of the input data.
        encoder_dims (List[int]): Dimensions of hidden layers in the encoder.
        latent_dim (int): Dimensionality of the latent space.
        normalizing_flow (NormalizingFlow): Sequence of flow transformations.
        decoder (nn.Sequential): Decoder network.
    """

    def __init__(
        self,
        input_dim: int,
        num_flows: int,
        encoder_dims: List[int],
        latent_dim: int,
        min_logvar: float = -20.0,
        max_logvar: float = 2.0,
    ):  # pylint: disable=too-many-positional-arguments
        """
        Initializes the VAE architecture.

        Args:
            input_dim (int): Input feature dimension.
            num_flows (int): Number of normalizing flows.
            encoder_dims (List[int]): List of hidden layer sizes for the encoder.
            latent_dim (int): Size of the latent representation.
        """

        super().__init__()

        if latent_dim <= 0:
            raise ValueError(f"Wrong embedding dimension: {latent_dim}")

        if len(encoder_dims) < 1:
            layer_dims_string = " ".join(str(dim) for dim in encoder_dims)
            raise ValueError(f"Wrong layer dimensions: {layer_dims_string}")

        if input_dim <= 0:
            raise ValueError(f"Wrong input dimension: {input_dim}")

        self.input_dim = input_dim
        self.encoder_dims = encoder_dims
        self.latent_dim = latent_dim
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

        dims = encoder_dims
        dims.insert(0, self.input_dim)

        # ---------------- Encoder ----------------
        self.encoder = nn.Sequential()
        if len(dims) > 1:
            for idx, layer_dim in enumerate(dims[:-1]):
                self.encoder.add_module(
                    f"encoder_linear_{idx}", nn.Linear(layer_dim, dims[idx + 1])
                )
                self.encoder.add_module(f"encoder_relu_{idx}", nn.ReLU(inplace=False))

        # Gaussian parameters
        self.mu_layer = nn.Linear(dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(dims[-1], latent_dim)

        # ---------------- Normalizing Flow ----------------
        flows = [PlanarFlow(latent_dim) for _ in range(num_flows)]
        self.normalizing_flow = NormalizingFlow(latent_dim, flows)

        # ---------------- Decoder ----------------
        self.decoder = nn.Sequential()
        self.decoder.add_module("latent_sample", nn.Linear(latent_dim, dims[-1]))

        reverse_encoder = list(reversed(dims))
        for idx, layer_dim in enumerate(reverse_encoder[:-1]):
            self.decoder.add_module(
                f"decoder_linear_{idx}", nn.Linear(layer_dim, reverse_encoder[idx + 1])
            )
            self.decoder.add_module(f"decoder_relu_{idx}", nn.ReLU(inplace=False))
        self.decoder.add_module("decoder_sigmoid", nn.Sigmoid())

        # dropout
        self.dropout = nn.Dropout(p=0.2)

    def encode(self, x):
        """
        Encodes input into latent distribution parameters.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            mu (Tensor): Mean of the latent Gaussian distribution.
            logvar (Tensor): Log-variance of the latent Gaussian distribution.
        """
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        return mu, logvar

    @staticmethod
    def _kl_safe(kl_raw: torch.Tensor, min_kl: float = 1e-9) -> torch.Tensor:
        """
        Lifts values below *min_kl* without blocking gradients.

        If kl_raw >= min_kl   -> unchanged
        If kl_raw <  min_kl   -> returns *min_kl* but keeps the original gradient.
        """
        return kl_raw + (kl_raw < min_kl).float() * (min_kl - kl_raw.detach())

    def _kl_divergence(  # pylint: disable=unused-argument,too-many-positional-arguments
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        eps: torch.Tensor,
        z0: torch.Tensor,
        zk: torch.Tensor,
        log_det: torch.Tensor,
        min_kl: float = 1e-9,
    ) -> torch.Tensor:
        """
        Compute D_KL( q_K(z|x)  ||  p(z) ) for a VAE equipped with K normalizing
        flows.  The prior is the standard Gaussian  N(0, I).

        Args
            mu : Tensor, shape [B, D]
                Mean of the base posterior q_0(z|x).
            logvar : Tensor, shape [B, D]
                Log-variance of the base posterior.
            z0 : Tensor, shape [B, D]
                One sample drawn from q_0(z|x) via the reparameterisation trick.
            zk : Tensor, shape [B, D]
                The same sample after passing through the K flow transformations.
            log_det : Tensor, shape [B]
                Cumulative log |det J_f| over the K flows.
            min_kl : float, optional
                Positive floor to avoid negative KL due to numerical noise.

        Returns
            kl : Tensor, shape [B]
                Sample-wise KL divergence, guaranteed to be >= min_kl while still
                propagating correct gradients.
        """
        # # Method 1
        # # 1) closed‑form KL between q_0 = N(mu, exp(logvar)) and the prior N(0, I)
        # kl_gauss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        # # 2) contribution of the flow’s Jacobian
        # flow_jac = -log_det
        # # 3) difference in log‑prior caused by the flow: ½(||z_K||² - ||z_0||²)
        # norm_diff = 0.5 * (zk.pow(2).sum(1) - z0.pow(2).sum(1))
        # kl_raw = kl_gauss + flow_jac + norm_diff  # shape [B]
        # print("\n methode 1 : ", kl_raw.mean())

        # # # Method 2
        # log_prior_zk = (torch.distributions.normal.Normal(0.0, 1.0).log_prob(zk).sum(dim=1))
        # log_prior_z0 = (torch.distributions.Normal(0.0, 1.0).log_prob(z0).sum(-1))  # p(z₀)
        # norm_diff = log_prior_z0 - log_prior_zk
        # kl_raw = kl_gauss + flow_jac + norm_diff
        # print("\n methode 2 : ", kl_raw.mean())

        # # # Method 3
        # log_q0_z0 = (
        #   torch.distributions.normal.Normal(
        #       mu,
        #       (0.5 * logvar).exp()
        #   )
        #   .log_prob(z0)
        #   .sum(dim=1)
        # )
        # log_prior_zk = (torch.distributions.normal.Normal(0.0, 1.0).log_prob(zk).sum(dim=1))
        # log_qk_zk = log_q0_z0 - log_det
        # kl_raw = log_qk_zk - log_prior_zk
        # print("\n methode 3 : ", kl_raw.mean())

        # Method 4
        log_q0 = -0.5 * ((eps**2) + LOG2PI + logvar).sum(dim=1)
        log_pz = -0.5 * ((zk**2) + LOG2PI).sum(dim=1)
        kl_raw = log_q0 - log_pz - log_det

        # return kl_raw
        return self._kl_safe(kl_raw, min_kl=min_kl)

    def sample_from_normal_distribution(self, mu, logvar):
        """
        Samples from a Gaussian distribution using the reparameterization trick.

        Args:
            mu (Tensor): Mean tensor.
            logvar (Tensor): Log-variance tensor.

        Returns:
            Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z0 = mu + eps * std
        return z0, eps

    def decode(self, z):
        """
        Decodes latent representation into reconstructed input.

        Args:
            z (Tensor): Latent vector.

        Returns:
            Tensor: Reconstructed input.
        """
        return self.decoder(z)

    def hidden_params(self, x):
        """
        Computes mean and standard deviation of latent distribution from input.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tuple[Tensor, Tensor]: Mean and standard deviation of latent Gaussian.
        """
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        std = torch.exp(0.5 * logvar)
        return mu, std

    def forward(self, x):
        """
        qK: density obtained by successively transforming
            a random variable z0 with distribution q0, through
            a chain of K transformations fk
        ln( qK(z_K) ) = ln ( q0(z0) ) - sum_{k=1..K} ln( |det( df_k/dz_{k-1} )| )
        with z_k = f_k( z_{k-1} )
        """

        # 1. Encoder : get latent distribution parameters µ and log σ²
        mu, logvar = self.encode(x.view(-1, self.input_dim))  # (B, D)
        z0, eps = self.sample_from_normal_distribution(mu, logvar)  # (B, D)
        zk, log_det_jakob = self.normalizing_flow(z0)  # (B, D) & (B,)
        kl_div = self._kl_divergence(mu, logvar, eps, z0, zk, log_det_jakob)
        x_hat = self.decode(zk)
        return x_hat, zk, kl_div


# Variational Inference with Normalizing Flows
# cf. https://arxiv.org/pdf/1505.05770.pdf


class Flow(nn.Module):
    """
    Base class for a normalizing flow transformation.
    """

    def __init__(self):
        """
        Initializes the base flow class.
        """
        nn.Module.__init__(self)

    def init_parameters(self):
        """
        Initializes parameters of the flow using uniform distribution.
        """
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)


class PlanarFlow(Flow):
    """
    Planar flow transformation for improving posterior expressiveness.

    Applies the transformation:
        f(z) = z + u * tanh(w · z + b)

    The Jacobian determinant is also computed for density estimation using the formula :
        jacobian(z) = I + transpose(u) . tanh'( transpose(w).z + b) . w
    """

    def __init__(self, dim):
        """
        Initializes the PlanarFlow transformation.

        Args:
            dim (int): Dimension of the latent space.
        """
        super().__init__()
        self.u = nn.Parameter(torch.Tensor(1, dim))
        self.w = nn.Parameter(torch.Tensor(1, dim))
        self.b = nn.Parameter(torch.Tensor(1))
        self.init_parameters()

    def forward(self, z):
        """
        Applies the planar flow transformation and computes the log-determinant Jacobian.

        Args:
            z (Tensor): Latent variable of shape (batch_size, dim).

        Returns:
            Tuple[Tensor, Tensor]:
                - Transformed latent variable,
                - Log absolute determinant of the Jacobian.
        """
        linear = F.linear(z, self.w, self.b)  # w·z + b ; pylint: disable=not-callable
        tanh = torch.tanh(linear)
        z_new = z + self.u * tanh

        psi = self.w * (1 - tanh**2)  # w ⊙ (1 − tanh²)
        det_j = 1 + torch.mm(psi, self.u.t())  # scalar per sample

        eps = 1e-6  # 1e-9
        logdet = torch.log(det_j.abs() + eps).squeeze(1)  # [B]
        return z_new, logdet


# Main class for normalizing flow
class NormalizingFlow(nn.Module):
    """
    Sequence of flow transformations applied to a latent variable.

    Attributes:
        flows (nn.ModuleList): List of flow modules.
        base_density (MultivariateNormal): Standard Gaussian distribution used as prior.
    """

    def __init__(self, dim, flows):
        """
        Initializes the normalizing flow model.

        Args:
            dim (int): Dimension of the latent space.
            flows (List[nn.Module]): List of flow transformations.
        """
        super().__init__()
        self.dim = dim
        self.flows = nn.ModuleList(flows)
        # TODO how is base_density used?
        self.base_density = torch.distributions.MultivariateNormal(
            torch.zeros(dim), torch.eye(dim)
        )

    def forward(self, z):
        """
        Applies the sequence of flows to the input latent variable.

        Args:
            z (Tensor): Latent variable sampled from approximate posterior.

        Returns:
            Tuple[Tensor, Tensor]:
                - Final transformed latent variable,
                - Sum of log-determinant Jacobians across flows.
        """
        log_det = torch.zeros(z.shape[0], device=z.device)
        logger.debug("inside log_det shape : %s", log_det.shape)
        # Applies series of flows
        for flow in self.flows:
            z, log_d = flow(z)
            logger.debug("inside z shape : %s", z.shape)
            logger.debug("inside log_d shape : %s", log_d.shape)
            log_det += log_d
        return z, log_det
