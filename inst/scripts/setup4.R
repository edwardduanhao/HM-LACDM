require(ggplot2)

Q3_mat <- as.matrix(read.table("inst/Q_Matrix/Q_3.txt"))
Q4_mat <- as.matrix(read.table("inst/Q_Matrix/Q_4.txt"))
N_dataset <- 50

# -------------------------- setup 4 -------------------------- #

# I = 1000, K = 4, indT = 3

# generate data

data4 <- data_generate(
  I = 1000, # number of respondents
  K = 4, # number of latent attributes
  J = 30, # number of items
  indT = 3, # number of time points
  N_dataset = N_dataset, # number of datasets to generate
  seed = 2025, # seed for reproducibility
  Q_mat = Q4_mat # generate a random Q-matrix if NULL
)

# run the HMLCDM algorithm

res4 <- vector(mode = "list", length = N_dataset)

for (i in seq(N_dataset)) {
  print(i)
  res4[[i]] <- HMLCDM_VB(
    data = list("Y" = data4$Y[i, , , ], "K" = data4$K, "ground_truth" = data4$ground_truth),
    max_iter = 100, # maximum number of iterations
    elbo = FALSE,
    alpha_level = 0.01
  )
}

runtime4 <- sapply(res4, \(x){
  x$runtime
})

mean(runtime4)

Q_acc4 <- sapply(res4, \(x) x$Q_acc)

mean(Q_acc4)

Q_FP4 <- sapply(res4, \(x) mean((x$Q_hat == 1) * (data4$ground_truth$Q_matrix == 0)))
mean(Q_FP4)

Q_FN4 <- sapply(res4, \(x) mean((x$Q_hat == 0) * (data4$ground_truth$Q_matrix == 1)))
mean(Q_FN4)

beta_rmse4 <- sapply(seq(N_dataset), \(i) (as.vector(res4[[i]]$beta - res4[[i]]$beta_true)[as.logical(as.vector(data4$ground_truth$Q_matrix))])^2 |>
  mean() |>
  sqrt())

pii_rmse4 <- sapply(seq(N_dataset), \(i) (res4[[i]]$pii - data4$ground_truth$pii)^2 |>
  mean() |>
  sqrt())

tau_rmse4 <- sapply(seq(N_dataset), \(i) (res4[[i]]$tau - data4$ground_truth$tau)^2 |>
  mean() |>
  sqrt())

fig_beta_rmse4 <- ggplot(data.frame(beta_rmse4), aes(x = beta_rmse4)) +
  geom_density(
    lwd = 1, colour = 2,
    fill = 2, alpha = 0.5
  ) +
  labs(x = expression(paste("RMSE of ", beta)), y = "Density") +
  theme_classic()

fig_pii_rmse4 <- ggplot(data.frame(pii_rmse4), aes(x = pii_rmse4)) +
  geom_density(
    lwd = 1, colour = 3,
    fill = 3, alpha = 0.5
  ) +
  labs(x = expression(paste("RMSE of ", pi)), y = "Density") +
  theme_classic()

fig_tau_rmse4 <- ggplot(data.frame(tau_rmse4), aes(x = tau_rmse4)) +
  geom_density(
    lwd = 1, colour = 4,
    fill = 4, alpha = 0.5
  ) +
  labs(x = expression(paste("RMSE of ", tau)), y = "Density") +
  theme_classic()

ggsave(
  filename = "inst/figures/multiplerun/setup4/beta_rmse.pdf", # Save in 'figures' subdirectory
  plot = fig_beta_rmse4,
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)

ggsave(
  filename = "inst/figures/multiplerun/setup4/pi_rmse.pdf", # Save in 'figures' subdirectory
  plot = fig_pii_rmse4,
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)

ggsave(
  filename = "inst/figures/multiplerun/setup4/tau_rmse.pdf", # Save in 'figures' subdirectory
  plot = fig_tau_rmse4,
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)
