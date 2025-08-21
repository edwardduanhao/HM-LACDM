library(ggplot2)

q3_mat <- as.matrix(read.table("inst/Q_Matrix/Q_3.txt"))
q4_mat <- as.matrix(read.table("inst/Q_Matrix/Q_4.txt"))
n_dataset <- 100

# -------------------------- setup 1 -------------------------- #

# i = 200, k = 3, t = 2

# generate data

data1 <- data_generate(
  i = 200, k = 3, j = 21, t = 2, n_dataset = n_dataset,
  seed = 2025, q_mat = q3_mat
)

# run the HMLCDM algorithm with multi-run approach

res1 <- vector(mode = "list", length = n_dataset)
multi_run_summaries <- vector(mode = "list", length = n_dataset)

for (i in seq(n_dataset)) {
  cat("=== Dataset", i, "of", n_dataset, "===\n")

  # Run multi-run HMLCDM for better reliability
  multi_result <- multi_run_hmlcdm(
    data = list(
      "y" = data1$y[i, , , ], "k" = data1$k, "ground_truth" = data1$ground_truth
    ),
    n_runs = 1, # Run 1 time per dataset
    max_iter = 100,
    alpha_level = 0.05,
    min_profile_threshold = 0.6, # Require at least 60% profile accuracy
    verbose = TRUE
  )

  # Store the best result
  res1[[i]] <- multi_result$best_result

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
runtime1 <- sapply(res1, \(x) x$runtime)
q_acc1 <- sapply(res1, \(x) x$Q_acc)
profile_acc1 <- sapply(res1, \(x) mean(x$profiles_acc))

# Print summary
cat("=== OVERALL RESULTS SUMMARY ===\n")
cat("Runtime (mean):", round(mean(runtime1), 2), "seconds\n")
cat("Q-matrix accuracy (mean):", round(mean(q_acc1), 3), "\n")
cat("Profile accuracy (mean):", round(mean(profile_acc1), 3), "\n")
cat(
  "Profile accuracy (range): [", round(min(profile_acc1), 3), ", ",
  round(max(profile_acc1), 3), "]\n"
)
cat("Datasets with profile accuracy > 0.8:", sum(profile_acc1 > 0.8),
    "out of", n_dataset, "\n")
cat("Datasets with profile accuracy < 0.4:", sum(profile_acc1 < 0.4),
    "out of", n_dataset, "\n")

q_fp1 <- sapply(res1, \(x) {
  mean((x$Q_hat == 1) * (data1$ground_truth$q_mat == 0))
})
mean(q_fp1)

q_fn1 <- sapply(res1, \(x) {
  mean((x$Q_hat == 0) * (data1$ground_truth$q_mat == 1))
})
mean(q_fn1)

beta_rmse1 <- sapply(seq(n_dataset), \(i) {
  (as.vector(res1[[i]]$beta - res1[[i]]$beta_true)[as.logical(
    as.vector(data1$ground_truth$q_mat)
  )])^2 |>
    mean() |>
    sqrt()
})

pii_rmse1 <- sapply(seq(n_dataset), \(i) {
  (res1[[i]]$pii - data1$ground_truth$pii)^2 |>
    mean() |>
    sqrt()
})

tau_rmse1 <- sapply(seq(n_dataset), \(i) {
  (res1[[i]]$tau - data1$ground_truth$tau)^2 |>
    mean() |>
    sqrt()
})

fig_beta_rmse1 <- ggplot(data.frame(beta_rmse1), aes(x = beta_rmse1)) +
  geom_density(
    lwd = 1, colour = 2,
    fill = 2, alpha = 0.5
  ) +
  labs(x = expression(paste("RMSE of ", beta)), y = "Density") +
  theme_classic()

fig_pii_rmse1 <- ggplot(data.frame(pii_rmse1), aes(x = pii_rmse1)) +
  geom_density(
    lwd = 1, colour = 3,
    fill = 3, alpha = 0.5
  ) +
  labs(x = expression(paste("RMSE of ", pi)), y = "Density") +
  theme_classic()

fig_tau_rmse1 <- ggplot(data.frame(tau_rmse1), aes(x = tau_rmse1)) +
  geom_density(
    lwd = 1, colour = 4,
    fill = 4, alpha = 0.5
  ) +
  labs(x = expression(paste("RMSE of ", tau)), y = "Density") +
  theme_classic()

ggsave(
  filename = "inst/figures/multiplerun/setup1/beta_rmse.pdf",
  plot = fig_beta_rmse1,
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)

ggsave(
  filename = "inst/figures/multiplerun/setup1/pi_rmse.pdf",
  plot = fig_pii_rmse1,
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)

ggsave(
  filename = "inst/figures/multiplerun/setup1/tau_rmse.pdf",
  plot = fig_tau_rmse1,
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)
