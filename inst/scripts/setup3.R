library(ggplot2)

q3_mat <- as.matrix(read.table("inst/Q_Matrix/Q_3.txt"))
q4_mat <- as.matrix(read.table("inst/Q_Matrix/Q_4.txt"))
n_dataset <- 100

# -------------------------- setup 3 -------------------------- #

# i = 200, k = 4, t = 2

# generate data

data3 <- data_generate(
  i = 200, k = 4, j = 30, t = 2, n_dataset = n_dataset,
  seed = 2026, q_mat = q4_mat
)

# run the HMLCDM algorithm with multi-run approach

res3 <- vector(mode = "list", length = n_dataset)
multi_run_summaries <- vector(mode = "list", length = n_dataset)

for (i in seq(n_dataset)) {
  cat("=== Dataset", i, "of", n_dataset, "===\n")

  # Run multi-run HMLCDM for better reliability
  multi_result <- multi_run_hmlcdm(
    data = list(
      "y" = data3$y[i, , , ], "k" = data3$k, "ground_truth" = data3$ground_truth
    ),
    n_runs = 1, # Run 1 time per dataset
    max_iter = 100,
    alpha_level = 0.05,
    min_profile_threshold = 0.6, # Require at least 60% profile accuracy
    verbose = TRUE
  )

  # Store the best result
  res3[[i]] <- multi_result$best_result

  # Store summary information
  multi_run_summaries[[i]] <- list(
    n_successful = multi_result$n_successful,
    n_good_profile = multi_result$n_good_profile,
    best_run_index = multi_result$best_run_index,
    metrics = multi_result$selection_metrics
  )

  cat("\n")
}

# Extract metrics from multi-run results
runtime3 <- sapply(res3, \(x) x$runtime)
q_acc3 <- sapply(res3, \(x) x$Q_acc)
profile_acc3 <- sapply(res3, \(x) mean(x$profiles_acc))

# Print summary
cat("=== OVERALL RESULTS SUMMARY ===\n")
cat("Runtime (mean):", round(mean(runtime3), 2), "seconds\n")
cat("Q-matrix accuracy (mean):", round(mean(q_acc3), 3), "\n")
cat("Profile accuracy (mean):", round(mean(profile_acc3), 3), "\n")
cat(
  "Profile accuracy (range): [", round(min(profile_acc3), 3), ", ",
  round(max(profile_acc3), 3), "]\n"
)
cat("Datasets with profile accuracy > 0.8:", sum(profile_acc3 > 0.8),
    "out of", n_dataset, "\n")
cat("Datasets with profile accuracy < 0.4:", sum(profile_acc3 < 0.4),
    "out of", n_dataset, "\n")

q_fp3 <- sapply(res3, \(x) {
  mean((x$Q_hat == 1) * (data3$ground_truth$q_mat == 0))
})
mean(q_fp3)

q_fn3 <- sapply(res3, \(x) {
  mean((x$Q_hat == 0) * (data3$ground_truth$q_mat == 1))
})
mean(q_fn3)

beta_rmse3 <- sapply(seq(n_dataset), \(i) {
  (as.vector(res3[[i]]$beta - res3[[i]]$beta_true)[as.logical(
    as.vector(data3$ground_truth$q_mat)
  )])^2 |>
    mean() |>
    sqrt()
})

pii_rmse3 <- sapply(seq(n_dataset), \(i) {
  (res3[[i]]$pii - data3$ground_truth$pii)^2 |>
    mean() |>
    sqrt()
})

tau_rmse3 <- sapply(seq(n_dataset), \(i) {
  (res3[[i]]$tau - data3$ground_truth$tau)^2 |>
    mean() |>
    sqrt()
})

beta_rmse3 <- beta_rmse3[beta_rmse3 <= 1]
pii_rmse3 <- pii_rmse3[beta_rmse3 <= 1]
tau_rmse3 <- tau_rmse3[beta_rmse3 <= 1]

fig_beta_rmse3 <- ggplot(data.frame(beta_rmse3), aes(x = beta_rmse3)) +
  geom_density(
    lwd = 1, colour = 2,
    fill = 2, alpha = 0.5
  ) +
  labs(x = expression(paste("RMSE of ", beta)), y = "Density") +
  theme_classic()

fig_pii_rmse3 <- ggplot(data.frame(pii_rmse3), aes(x = pii_rmse3)) +
  geom_density(
    lwd = 1, colour = 3,
    fill = 3, alpha = 0.5
  ) +
  labs(x = expression(paste("RMSE of ", pi)), y = "Density") +
  theme_classic()

fig_tau_rmse3 <- ggplot(data.frame(tau_rmse3), aes(x = tau_rmse3)) +
  geom_density(
    lwd = 1, colour = 4,
    fill = 4, alpha = 0.5
  ) +
  labs(x = expression(paste("RMSE of ", tau)), y = "Density") +
  theme_classic()

ggsave(
  filename = "inst/figures/multiplerun/setup3/beta_rmse.pdf",
  plot = fig_beta_rmse3,
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)

ggsave(
  filename = "inst/figures/multiplerun/setup3/pi_rmse.pdf",
  plot = fig_pii_rmse3,
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)

ggsave(
  filename = "inst/figures/multiplerun/setup3/tau_rmse.pdf",
  plot = fig_tau_rmse3,
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)
