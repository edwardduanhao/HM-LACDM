library(ggplot2)

# i <- 200
# k <- 3
# t <- 2
# seed <- 42
# signal <- "weak"

run_experiment <- function(i, k, t, signal, n_dataset = 100, seed = 42) {
  if (k == 3) {
    q_mat <- as.matrix(read.table("inst/Q_Matrix/Q_3.txt"))
  } else if (k == 4) {
    q_mat <- as.matrix(read.table("inst/Q_Matrix/Q_4.txt"))
  } else {
    stop("k must be either 3 or 4")
  }

  j <- nrow(q_mat)

  # Create file name based on simulation parameters
  folder_name <- paste0("i_", i, "_k_", k, "_t_", t, "_j_", j, "_signal_", signal)

  # Create directory if it doesn't exist
  path_fig <- "inst/out/figures/multiplerun/"
  path_table <- "inst/out/tables/multiplerun/"
  if (!dir.exists(file.path(path_fig, folder_name))) {
    dir.create(file.path(path_fig, folder_name), recursive = TRUE)
  }
  path_fig <- file.path(path_fig, folder_name)

  if (!dir.exists(file.path(path_table, folder_name))) {
    dir.create(file.path(path_table, folder_name), recursive = TRUE)
  }
  path_table <- file.path(path_table, folder_name)

  # Generate data
  data <- data_generate(
    i = i,
    k = k,
    j = j,
    t = t,
    n_dataset = n_dataset,
    seed = seed,
    q_mat = q_mat,
    signal = signal,
    device = "cpu"
  )

  # Run coordinate ascent variational inference for HMLCDM
  res <- vector("list", n_dataset)
  res_1 <- vector("list", n_dataset)

  for (iter_ in seq(n_dataset)) {
    cat("Running dataset", iter_, "\n")
    res[[iter_]] <- hmlcdm_vb(
      data = list(
        "y" = data$y[iter_, , , ],
        "k" = data$k,
        "ground_truth" = data$ground_truth
      ),
      max_iter = 100,
      elbo = FALSE,
      device = "cpu"
    )
    res[[iter_]] <- post_hoc(res[[iter_]], data, alpha_level = 0.05, q_mat_true = q_mat)
    res_1[[iter_]] <- post_hoc(res[[iter_]], data, alpha_level = 0.01, q_mat_true = q_mat)
  }

  # Runtime
  runtime <- sapply(res, \(x) x$runtime)

  # Profile accuracy
  profile_acc <- sapply(res, \(x) mean(x$profiles_acc))

  # Q-matrix recovery (accuracy)
  q_acc <- sapply(res, \(x) x$q_mat_acc)
  q_acc_1 <- sapply(res_1, \(x) x$q_mat_acc)

  # Q-matrix recovery (false positive rate)
  q_fp <- sapply(res, \(x) {
    mean((x$q_mat_hat == 1) * (data$ground_truth$q_mat == 0))
  })

  q_fp_1 <- sapply(res_1, \(x) {
    mean((x$q_mat_hat == 1) * (data$ground_truth$q_mat == 0))
  })

  # Q-matrix recovery (false negative rate)
  q_fn <- sapply(res, \(x) {
    mean((x$q_mat_hat == 0) * (data$ground_truth$q_mat == 1))
  })

  q_fn_1 <- sapply(res_1, \(x) {
    mean((x$q_mat_hat == 0) * (data$ground_truth$q_mat == 1))
  })

  beta_rmse <- sapply(seq(n_dataset), \(i) {
    (as.vector(res[[i]]$beta - res[[i]]$beta_true)[as.logical(
      as.vector(data$ground_truth$q_mat)
    )])^2 |>
      mean() |>
      sqrt()
  })

  pii_rmse <- sapply(seq(n_dataset), \(i) {
    (res[[i]]$pii - data$ground_truth$pii)^2 |>
      mean() |>
      sqrt()
  })

  tau_rmse <- sapply(seq(n_dataset), \(i) {
    (res[[i]]$tau - data$ground_truth$tau)^2 |>
      mean() |>
      sqrt()
  })

  plot_density(
    beta_rmse,
    color = 2,
    path = path_fig,
    file_name = "beta_rmse.pdf",
    xlab = expression(paste("RMSE of ", beta))
  )

  plot_density(
    pii_rmse,
    color = 3,
    path = path_fig,
    file_name = "pii_rmse.pdf",
    xlab = expression(paste("RMSE of ", pi))
  )

  plot_density(
    tau_rmse,
    color = 4,
    path = path_fig,
    file_name = "tau_rmse.pdf",
    xlab = expression(paste("RMSE of ", tau))
  )

  summary_table <- data.frame(
    Metric = c(
      "Runtime (seconds)",
      "Profile accuracy",
      "Q-matrix accuracy (0.05)",
      "Q-matrix false positive rate (0.05)",
      "Q-matrix false negative rate (0.05)",
      "Q-matrix accuracy (0.01)",
      "Q-matrix false positive rate (0.01)",
      "Q-matrix false negative rate (0.01)",
      "Beta RMSE",
      "Pii RMSE",
      "Tau RMSE"
    ),
    Mean = c(
      mean(runtime),
      mean(profile_acc),
      mean(q_acc),
      mean(q_fp),
      mean(q_fn),
      mean(q_acc_1),
      mean(q_fp_1),
      mean(q_fn_1),
      mean(beta_rmse),
      mean(pii_rmse),
      mean(tau_rmse)
    ),
    SD = c(
      sd(runtime),
      sd(profile_acc),
      sd(q_acc),
      sd(q_fp),
      sd(q_fn),
      sd(q_acc_1),
      sd(q_fp_1),
      sd(q_fn_1),
      sd(beta_rmse),
      sd(pii_rmse),
      sd(tau_rmse)
    )
  )

  write.csv(summary_table, file = file.path(path_table, "summary_table.csv"), row.names = FALSE)
}


i_vec <- c(200, 1000)
k_vec <- c(3, 4)
t_vec <- c(2, 3, 5)
signal_vec <- c("weak", "moderate", "strong")

for (i in i_vec) {
  for (k in k_vec) {
    for (t in t_vec) {
      for (signal in signal_vec) {
        cat("Running experiment with i =", i, ", k =", k, ", t =", t, ", signal =", signal, "\n")
        run_experiment(i = i, k = k, t = t, signal = signal, n_dataset = 100, seed = 42)
      }
    }
  }
}
