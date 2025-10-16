# HM-LACDM 0.1.0

## Initial Release

This is the first public release of the HM-LACDM package.

### Features

* **Variational Inference Algorithm**: Implemented fast variational Bayes inference for Hidden Markov Longitudinal Additive Cognitive Diagnostic Models
* **GPU Acceleration**: Full support for CPU and CUDA-enabled GPU computation via torch backend
* **Data Generation**: `data_generate()` function for simulating longitudinal CDM data
* **Model Fitting**: `hmlcdm_vb()` as the main model estimation function
* **Visualization Tools**: Comprehensive plotting functions for:
  - Parameter recovery analysis (`plot_beta_recovery()`, `plot_pii_recovery()`, `plot_tau_recovery()`)
  - Convergence diagnostics (`plot_elbo_trace()`)
  - Parameter traces (`plot_alpha_trace()`, `plot_beta_trace()`, `plot_omega_trace()`)
  - Model diagnostics (`plot_density()`, `q_mat_recovery()`)
* **Real Data Examples**: Included `realdata` and `realdata1` datasets for demonstration
* **Example Scripts**: Comprehensive analysis scripts in `inst/scripts/`:
  - `singlerun.R` - Single model estimation
  - `multipleruns.R` - Monte Carlo simulation studies
  - `realdata.R` - Real data analysis

### Documentation

* Complete function documentation using roxygen2
* MIT License
* GitHub repository: https://github.com/edwardduanhao/HMLCDM

### Dependencies

* R >= 4.1.0
* torch (PyTorch for R)
* tmvtnorm
* zeallot
* iterpc
* cli

---

*Release Date: 2024*
