require(ggplot2)

Q3_mat <- as.matrix(read.table("inst/Q_Matrix/Q_3.txt"))
Q4_mat <- as.matrix(read.table("inst/Q_Matrix/Q_4.txt"))
N_dataset <- 50

# -------------------------- setup 3 -------------------------- #

# I = 200, K = 4, indT = 3

# generate data

data3 <- data_generate(
  I = 200, # number of respondents
  K = 4, # number of latent attributes
  J = 30, # number of items
  indT = 3, # number of time points
  N_dataset = N_dataset, # number of datasets to generate
  seed = 2026, # seed for reproducibility
  Q_mat = Q4_mat # generate a random Q-matrix if NULL
)

# run the HMLCDM algorithm

res3 <- vector(mode = "list", length = N_dataset)

for (i in seq(N_dataset)) {
  print(i)
  res3[[i]] <- HMLCDM_VB(
    data = list("Y" = data3$Y[i, , , ], "K" = data3$K, "ground_truth" = data3$ground_truth),
    max_iter = 100, # maximum number of iterations
    elbo = FALSE,
    alpha_level = 0.05
  )
}

runtime3 <- sapply(res3, \(x){
  x$runtime
})

mean(runtime3)

Q_acc3 <- sapply(res3, \(x) x$Q_acc)

mean(Q_acc3)

Q_FP3 <- sapply(res3, \(x) mean((x$Q_hat == 1) * (data3$ground_truth$Q_matrix == 0)))
mean(Q_FP3)

Q_FN3 <- sapply(res3, \(x) mean((x$Q_hat == 0) * (data3$ground_truth$Q_matrix == 1)))
mean(Q_FN3)

beta_rmse3 <- sapply(seq(N_dataset), \(i) (as.vector(res3[[i]]$beta - res3[[i]]$beta_true)[as.logical(as.vector(data3$ground_truth$Q_matrix))])^2 |>
  mean() |>
  sqrt())

pii_rmse3 <- sapply(seq(N_dataset), \(i) (res3[[i]]$pii - data3$ground_truth$pii)^2 |>
  mean() |>
  sqrt())

tau_rmse3 <- sapply(seq(N_dataset), \(i) (res3[[i]]$tau - data3$ground_truth$tau)^2 |>
  mean() |>
  sqrt())

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
  filename = "inst/figures/multiplerun/setup3/beta_rmse.pdf", # Save in 'figures' subdirectory
  plot = fig_beta_rmse3,
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)

ggsave(
  filename = "inst/figures/multiplerun/setup3/pi_rmse.pdf", # Save in 'figures' subdirectory
  plot = fig_pii_rmse3,
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)

ggsave(
  filename = "inst/figures/multiplerun/setup3/tau_rmse.pdf", # Save in 'figures' subdirectory
  plot = fig_tau_rmse3,
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)
