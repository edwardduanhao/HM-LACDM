require(ggplot2)

Q3_mat <- as.matrix(read.table("inst/Q_Matrix/Q_3.txt"))
Q4_mat <- as.matrix(read.table("inst/Q_Matrix/Q_4.txt"))
N_dataset <- 50

# -------------------------- setup 2 -------------------------- #

# I = 1000, K = 3, indT = 2

# generate data

data2 <- data_generate(
  I = 1000, # number of respondents
  K = 3, # number of latent attributes
  J = 21, # number of items
  indT = 2, # number of time points
  N_dataset = N_dataset, # number of datasets to generate
  seed = 2025, # seed for reproducibility
  Q_mat = Q3_mat # generate a random Q-matrix if NULL
)

# run the HMLCDM algorithm

res2 <- vector(mode = "list", length = N_dataset)

for (i in seq(N_dataset)) {
  print(i)
  res2[[i]] <- HMLCDM_VB(
    data = list("Y" = data2$Y[i, , , ], "K" = data2$K, "ground_truth" = data2$ground_truth),
    max_iter = 100, # maximum number of iterations
    elbo = FALSE,
    alpha_level = 0.01
  )
}

runtime2 <- sapply(res2, \(x){
  x$runtime
})

mean(runtime2)

Q_acc2 <- sapply(res2, \(x) x$Q_acc)

mean(Q_acc2)

Q_FP2 <- sapply(res2, \(x) mean((x$Q_hat == 1) * (data2$ground_truth$Q_matrix == 0)))
mean(Q_FP2)

Q_FN2 <- sapply(res2, \(x) mean((x$Q_hat == 0) * (data2$ground_truth$Q_matrix == 1)))
mean(Q_FN2)

beta_rmse2 <- sapply(seq(N_dataset), \(i) (as.vector(res2[[i]]$beta - res2[[i]]$beta_true)[as.logical(as.vector(data2$ground_truth$Q_matrix))])^2 |>
  mean() |>
  sqrt())

pii_rmse2 <- sapply(seq(N_dataset), \(i) (res2[[i]]$pii - data2$ground_truth$pii)^2 |>
  mean() |>
  sqrt())

tau_rmse2 <- sapply(seq(N_dataset), \(i) (res2[[i]]$tau - data2$ground_truth$tau)^2 |>
  mean() |>
  sqrt())

fig_beta_rmse2 <- ggplot(data.frame(beta_rmse2), aes(x = beta_rmse2)) +
  geom_density(
    lwd = 1, colour = 2,
    fill = 2, alpha = 0.5
  ) +
  labs(x = expression(paste("RMSE of ", beta)), y = "Density") +
  theme_classic()

fig_pii_rmse2 <- ggplot(data.frame(pii_rmse2), aes(x = pii_rmse2)) +
  geom_density(
    lwd = 1, colour = 3,
    fill = 3, alpha = 0.5
  ) +
  labs(x = expression(paste("RMSE of ", pi)), y = "Density") +
  theme_classic()

fig_tau_rmse2 <- ggplot(data.frame(tau_rmse2), aes(x = tau_rmse2)) +
  geom_density(
    lwd = 1, colour = 4,
    fill = 4, alpha = 0.5
  ) +
  labs(x = expression(paste("RMSE of ", tau)), y = "Density") +
  theme_classic()

ggsave(
  filename = "inst/figures/multiplerun/setup2/beta_rmse.pdf", # Save in 'figures' subdirectory
  plot = fig_beta_rmse2,
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)

ggsave(
  filename = "inst/figures/multiplerun/setup2/pi_rmse.pdf", # Save in 'figures' subdirectory
  plot = fig_pii_rmse2,
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)

ggsave(
  filename = "inst/figures/multiplerun/setup2/tau_rmse.pdf", # Save in 'figures' subdirectory
  plot = fig_tau_rmse2,
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)
