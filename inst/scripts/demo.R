library(ggplot2)

# load the Q-matrix

Q_mat <- as.matrix(read.table("inst/Q_Matrix/Q_3.txt"))

# generate data

data <- data_generate(
  I = 200, # number of respondents
  K = 3, # number of latent attributes
  J = 21, # number of items
  indT = 2, # number of time points
  N_dataset = 1, # number of datasets to generate
  seed = 2025, # seed for reproducibility
  Q_mat = Q_mat # generate a random Q-matrix if NULL
)

# run the HMLCDM algorithm

res <- HMLCDM_VB(
  data = data,
  max_iter = 100, # maximum number of iterations
  elbo = TRUE
)


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
# scale_color_brewer(palette = "Set1")


# ------------------------------ omega trace ------------------------------ #


df_omega <- matrix(res$omega_trace, nrow = dim(res$omega_trace)[1], ncol = prod(dim(res$omega_trace)[-1]))[seq(iter_trunc), ] |>
  as.data.frame() |>
  stack()

df_omega$iter <- rep(seq(iter_trunc), times = ncol(res$omega_trace))

names(df_omega) <- c("value", "variable", "iter")

fig_omega <- ggplot(df_omega, aes(x = iter, y = value, group = variable)) +
  geom_line(color = "#2774AE") +
  theme_classic() +
  theme(legend.position = "none") +
  labs(x = "Iteration", y = expression(omega))
# scale_color_brewer(palette = "Set1")


# ------------------------------ beta trace ------------------------------ #


df_beta <- matrix(res$beta_trace, nrow = dim(res$beta_trace)[1], ncol = prod(dim(res$beta_trace)[-1]))[seq(iter_trunc), ] |>
  as.data.frame() |>
  stack()

df_beta$iter <- rep(seq(iter_trunc), times = ncol(res$beta_trace))

df_beta$type <- rep(as.factor(res$beta_true), each = iter_trunc)

names(df_beta) <- c("value", "variable", "iter", "type")

fig_beta <- ggplot(df_beta, aes(x = iter, y = value, group = variable, color = type)) +
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
  filename = "inst/figures/singlerun/beta_trace.pdf", # Save in 'figures' subdirectory
  plot = fig_beta,
  width = 8,
  height = 6,
  units = "in",
  dpi = 600
)

ggsave(
  filename = "inst/figures/singlerun/omega_trace.pdf", # Save in 'figures' subdirectory
  plot = fig_omega,
  width = 8,
  height = 6,
  units = "in",
  dpi = 600
)

ggsave(
  filename = "inst/figures/singlerun/alpha_trace.pdf", # Save in 'figures' subdirectory
  plot = fig_alpha,
  width = 8,
  height = 6,
  units = "in",
  dpi = 600
)
