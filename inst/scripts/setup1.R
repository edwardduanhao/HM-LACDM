q3_mat <- as.matrix(read.table("inst/Q_Matrix/Q_3.txt"))
q4_mat <- as.matrix(read.table("inst/Q_Matrix/Q_4.txt"))
n_dataset <- 20

# -------------------------- setup 1 -------------------------- #

# I = 200, K = 3, indT = 2

# generate data

data1 <- data_generate(
  i = 500, k = 3, j = 21, t = 2, n_dataset = n_dataset,
  seed = 2025, q_mat = q3_mat
)

# run the HMLCDM algorithm

res1 <- vector(mode = "list", length = n_dataset)

for (i in seq(n_dataset)) {
  print(i)
  res1[[i]] <- hmlcdm_vb(
    data = list(
      "y" = data1$y[i, , , ], "k" = data1$k, "ground_truth" = data1$ground_truth
    ),
    max_iter = 100, # maximum number of iterations
    elbo = FALSE,
    alpha_level = 0.05
  )
}

runtime1 <- sapply(res1, \(x) x$runtime)

mean(runtime1)

q_acc1 <- sapply(res1, \(x) x$Q_acc)

mean(q_acc1)

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
