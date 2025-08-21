library(ggplot2)

# load the Q-matrix

q_mat <- as.matrix(read.table("inst/Q_Matrix/Q_3.txt"))

# generate data

data <- data_generate(
  i = 200, k = 3, j = 21, t = 2, n_dataset = 1, seed = 2025, q_mat = q_mat
)

# run the HMLCDM algorithm with multi-run approach for better reliability

# Use multi-run approach to get the best result
multi_result <- multi_run_hmlcdm(
  data = data,
  n_runs = 1,  # Run 3 times and pick the best
  max_iter = 100,
  min_profile_threshold = 0.7,  # Higher threshold for demo
  verbose = FALSE
)

# Use the best result for plotting
res <- multi_result$best_result

# Print selection summary
cat("\n=== DEMO MULTI-RUN SUMMARY ===\n")
cat("Profile accuracy of selected run:", round(mean(res$profiles_acc), 3), "\n")
cat("Q-matrix accuracy:", round(res$Q_acc, 3), "\n")
cat("Final ELBO:", round(tail(res$elbo, 1), 1), "\n")
compare_runs(multi_result)

# ------------------------------ alpha trace ------------------------------ #

iter_trunc <- 30

df_alpha <- as.data.frame(res$alpha_trace[seq(iter_trunc), ]) |> stack()

df_alpha$iter <- rep(seq(iter_trunc), times = ncol(res$alpha_trace))

names(df_alpha) <- c("value", "variable", "iter")

fig_alpha <- ggplot(df_alpha, aes(x = iter, y = value, group = variable)) +
  geom_line(color = "#2774AE") +
  theme_classic() +
  theme(legend.position = "none") +
  labs(x = "Iteration", y = expression(alpha))


# ------------------------------ omega trace ------------------------------ #


df_omega <- matrix(res$omega_trace,
  nrow = dim(res$omega_trace)[1],
  ncol = prod(dim(res$omega_trace)[-1])
)[seq(iter_trunc), ] |>
  as.data.frame() |>
  stack()

df_omega$iter <- rep(seq(iter_trunc), times = ncol(res$omega_trace))

names(df_omega) <- c("value", "variable", "iter")

fig_omega <- ggplot(df_omega, aes(x = iter, y = value, group = variable)) +
  geom_line(color = "#2774AE") +
  theme_classic() +
  theme(legend.position = "none") +
  labs(x = "Iteration", y = expression(omega))


# ------------------------------ beta trace ------------------------------ #


df_beta <- matrix(res$beta_trace,
  nrow = dim(res$beta_trace)[1],
  ncol = prod(dim(res$beta_trace)[-1])
)[seq(iter_trunc), ] |>
  as.data.frame() |>
  stack()

df_beta$iter <- rep(seq(iter_trunc), times = ncol(res$beta_trace))

df_beta$type <- rep(as.factor(res$beta_true), each = iter_trunc)

names(df_beta) <- c("value", "variable", "iter", "type")

fig_beta <- ggplot(df_beta,
                   aes(x = iter, y = value, group = variable, color = type)) +
  geom_line() +
  theme_classic() +
  theme(legend.position = "none") +
  labs(x = "Iteration", y = expression(beta)) +
  scale_color_brewer(palette = "Set1")


# save the figures

if (!dir.exists("inst/figures/singlerun")) {
  dir.create("inst/figures/singlerun", recursive = TRUE)
}

ggsave(
  filename = "inst/figures/singlerun/beta_trace_large.pdf",
  plot = fig_beta,
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)

ggsave(
  filename = "inst/figures/singlerun/omega_trace_large.pdf",
  plot = fig_omega,
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)

ggsave(
  filename = "inst/figures/singlerun/alpha_trace_large.pdf",
  plot = fig_alpha,
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)

df_elbo <- data.frame(
  iteration = seq(iter_trunc),
  elbo = res$elbo[seq(iter_trunc)]
)

fig_elbo <- ggplot(df_elbo, aes(x = iteration, y = elbo)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  labs(x = "Iteration", y = "Evidence Lower Bound") +
  theme_classic()


ggsave(
  filename = "inst/figures/singlerun/elbo_trace_large.pdf",
  plot = fig_elbo,
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)

df_tau_recovery <- data.frame(
  tau_true = as.vector(data$ground_truth$tau),
  tau_hat = as.vector(res$tau)
)

fig_tau_recovery <- ggplot(df_tau_recovery, aes(x = tau_true, y = tau_hat)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  theme_minimal() +
  labs(
    x = expression(paste("True ", tau)),
    y = expression(paste("Estimated ", tau))
  )

ggsave(
  filename = "inst/figures/singlerun/tau_recovery_large.pdf",
  plot = fig_tau_recovery,
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)


df_beta_recovery <- data.frame(
  beta_true = as.vector(res$beta_true),
  beta_hat = as.vector(res$beta)
)

fig_beta_recovery <- ggplot(df_beta_recovery,
                            aes(x = beta_true, y = beta_hat)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  theme_minimal() +
  labs(
    x = expression(paste("True ", beta)),
    y = expression(paste("Estimated ", beta))
  )

ggsave(
  filename = "inst/figures/singlerun/beta_recovery_large.pdf",
  plot = fig_beta_recovery,
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)


df_pii_recovery <- data.frame(
  pii_true = as.vector(data$ground_truth$pii),
  pii_hat = as.vector(res$pii)
)

fig_pii_recovery <- ggplot(df_pii_recovery, aes(x = pii_true, y = pii_hat)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  theme_minimal() +
  labs(
    x = expression(paste("True ", pi)),
    y = expression(paste("Estimated ", pi))
  )

ggsave(
  filename = "inst/figures/singlerun/pii_recovery_large.pdf",
  plot = fig_pii_recovery,
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)


res$Q_hat

res$Q_acc
