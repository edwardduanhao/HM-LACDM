library(ggplot2)

# Load the Q-matrix
q_mat <- as.matrix(read.table("inst/Q_Matrix/Q_3.txt"))

# Generate data
data <- data_generate(
  i = 500, # 10000
  k = 3,
  j = 21,
  t = 2,
  n_dataset = 1,
  seed = 42,
  q_mat = q_mat,
  device = "cpu"
)

# Run coordinate ascent variational inference for HMLCDM
res <- hmlcdm_vb(
  data = data,
  max_iter = 100,
  elbo = TRUE,
  device = "cpu"
)

# Run post-hoc analysis
res <- post_hoc(res, data, alpha_level = 0.05, q_mat_true = data$ground_truth$q_mat)

cat("Q-matrix recovery accuracy: ", res$q_mat_acc, "\n")

# Make plots
path <- "inst/figures/singlerun/small"
# path <- "inst/figures/singlerun/large"

if (!dir.exists(path)) {
  dir.create(path, recursive = TRUE)
}

plot_alpha_trace(res$alpha_trace, path = path)

plot_beta_trace(res$beta_trace, res$beta_true, path = path)

plot_omega_trace(res$omega_trace, path = path)

plot_pii_recovery(res$pii, data$ground_truth$pii, path = path)

plot_beta_recovery(res$beta, res$beta_true, path = path)

plot_tau_recovery(res$tau, data$ground_truth$tau, path = path)

plot_elbo_trace(res$elbo, path = path)
