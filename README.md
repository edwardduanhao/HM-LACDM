# HM-LACDM

<!-- badges: start -->
[![R-CMD-check](https://img.shields.io/badge/R%20CMD%20check-passing-brightgreen)](https://github.com/edwardduanhao/HMLCDM)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![R version](https://img.shields.io/badge/R-%E2%89%A5%204.1.0-blue)](https://www.r-project.org/)
<!-- badges: end -->

> **A Hidden Markov Longitudinal Additive Cognitive Diagnostic Model implemented in R with torch backend and fast Variational Inference.**

## Overview

`HM-LACDM` implements a variational inference algorithm for Hidden Markov Longitudinal Cognitive Diagnostic Models (HMLCDM). This package enables efficient Bayesian estimation of cognitive diagnostic models with temporal dependencies, making it ideal for analyzing longitudinal educational assessment data.

### Key Features

- **Variational Bayesian Inference**: Fast, scalable approximate inference using variational methods
- **GPU Acceleration**: Powered by [torch](https://torch.mlverse.org/) for efficient computation on CPU or CUDA-enabled GPUs
- **Longitudinal Modeling**: Captures temporal dynamics in latent attribute mastery through Hidden Markov structures
- **Cognitive Diagnostic Modeling**: Additive Cognitive Diagnostic Model (LCDM) framework for fine-grained skill assessment
- **Comprehensive Diagnostics**: Built-in functions for parameter recovery analysis, trace plots, and model evaluation

## Installation

### From GitHub

You can install the development version from GitHub:

```r
# Install devtools if not already installed
if (!requireNamespace("devtools", quietly = TRUE)) {
  install.packages("devtools")
}

# Install HM-LACDM
devtools::install_github("edwardduanhao/HMLCDM")
```

### Dependencies

The package requires R >= 4.1.0 and depends on:

- `torch` - Deep learning framework (PyTorch for R)
- `tmvtnorm` - Truncated multivariate normal distributions
- `zeallot` - Multiple assignment operator
- `iterpc` - Iterators for permutations and combinations
- `cli` - Enhanced command-line interface

**Note**: First-time `torch` users will need to install LibTorch. This happens automatically when you first load the package:

```r
library(HM-LACDM)  # Will prompt for LibTorch installation if needed
```

## Quick Start

### Generate Synthetic Data

```r
library(HM-LACDM)

# Generate data: 200 examinees, 3 attributes, 25 items, 2 time points
sim_data <- data_generate(
  i = 200,    # Number of examinees
  k = 3,      # Number of attributes
  j = 25,     # Number of items
  t = 2,      # Number of time points
  seed = 42
)
```

### Fit the Model

```r
# Run variational Bayes inference
results <- hmlcdm_vb(
  data = sim_data,
  max_iter = 100,
  elbo = TRUE,         # Track Evidence Lower Bound
  device = "auto"      # Use GPU if available
)
```

### Visualize Results

```r
# Plot parameter recovery
plot_beta_recovery(results, sim_data$ground_truth$beta)
plot_pii_recovery(results, sim_data$ground_truth$pii)
plot_tau_recovery(results, sim_data$ground_truth$tau)

# Plot convergence
plot_elbo_trace(results)

# Plot parameter traces
plot_alpha_trace(results)
plot_beta_trace(results)
```

## Model Description

The Hidden Markov Longitudinal Additive Cognitive Diagnostic Model (HM-LACDM) combines:

1. **Cognitive Diagnostic Model (CDM)**: Models the relationship between observed item responses and latent attribute mastery patterns
2. **Hidden Markov Model (HMM)**: Captures transitions in attribute mastery over time
3. **Variational Inference**: Provides fast approximate posterior inference for model parameters

### Model Parameters

- **β (beta)**: Item parameters (intercepts and main effects)
- **π (pii)**: Initial distribution of attribute profiles at time 1
- **τ (tau)**: Transition probabilities between attribute profiles across time
- **α (alpha)**: Individual attribute mastery profiles over time

## Real Data Analysis

The package includes real dataset examples:

```r
# Load real data
data("realdata", package = "HM-LACDM")

# Fit model to real data
results_real <- hmlcdm_vb(realdata, max_iter = 150)
```

## Example Scripts

The `inst/scripts/` directory contains comprehensive examples:

- `singlerun.R` - Single model estimation with visualization
- `multipleruns.R` - Monte Carlo simulations across parameter configurations
- `realdata.R` - Real data analysis workflow

## GPU Support

To leverage GPU acceleration:

```r
# Explicitly request CUDA device
results <- hmlcdm_vb(data, device = "cuda")

# Check if CUDA is available
torch::torch_cuda_is_available()
```

## Citation

If you use this package in your research, please cite:

```
Duan, H. (2024). HM-LACDM: Variational Inference Algorithm for Hidden Markov
  Longitudinal Additive Cognitive Diagnostic Models. R package version 0.1.0.
  https://github.com/edwardduanhao/HMLCDM
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact

**Hao Duan**
Email: hduan7@ucla.edu
GitHub: [@edwardduanhao](https://github.com/edwardduanhao)

## Links

- **GitHub Repository**: https://github.com/edwardduanhao/HMLCDM
- **Bug Reports**: https://github.com/edwardduanhao/HMLCDM/issues

---

