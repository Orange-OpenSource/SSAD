# SSAD — Self-Supervised Anomaly Detection Library

A Python library for autoencoder-based **anomaly detection** with self-supervised training and dynamic per-sample **confidence** updates.

## Key Features

- Compute per-sample anomaly scores
- Estimate confidence from score distributions
- Recalibrate confidence intervals during training
- Apply confidence-aware losses (normal / abnormal / uncertain)
- Track experiments and artifacts with **MLflow**

## Installation

```bash
pip install ssad
```

For development setup:

```bash
pip install -e .[dev]
```

## Quick Links

- **Repository**: https://github.com/Orange-OpenSource/SSAD
- **Examples**: https://github.com/Orange-OpenSource/SSAD/tree/main/examples
- **Issues**: https://github.com/Orange-OpenSource/SSAD/issues


## References

1. N. Najari, S. Berlemont, G. Lefebvre, S. Duffner, C. Garcia,  
   *Robust Variational Autoencoders and Normalizing Flows for Unsupervised Network Anomaly Detection*,  
   AINA 2022, doi: 10.1007/978-3-030-99587-4_24

2. N. Najari, S. Berlemont, G. Lefebvre, S. Duffner, C. Garcia,  
   *RADON: Robust Autoencoder for Unsupervised Anomaly Detection*,  
   SIN 2021, doi: 10.1109/SIN54109.2021.9699174
