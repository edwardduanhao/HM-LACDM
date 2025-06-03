source("utils.R")
source("utils_vb.R")

HMLCDM_VB <- function(data,
                      max_iter = 100,
                      alpha_level = 0.05,
                      elbo = FALSE # set to TRUE to compute the ELBO at each iteration
) {
  # print the number of threads Torch is using

  print(paste("Torch is using", torch_get_num_threads(), "threads"))

  # setup a timer

  start_time <- Sys.time()

  # --------------------- Retrieve and format data --------------------- #

  # response matrix Y, shape (N, indT, J)

  Y <- torch_tensor(data$Y)

  # number of observations, number of time points, number of items

  c(I, indT, J) %<-% Y$shape

  # number of attributes

  K <- data$K

  L <- 2^K

  # Generate the delta matrix for each item

  Delta_matrices <- lapply(seq(J), \(j) build_delta(
    q = rep(1, K),
    interact = FALSE
  ) |>
    torch_tensor())

  beta_dim <- rep(K + 1, J)

  # --------------------- Parameter initialization --------------------- #

  # beta

  M_beta_prior <- lapply(beta_dim, \(d) torch_zeros(d)) # Mean

  M_beta <- lapply(beta_dim, \(d) torch_tensor(c(-3, rep(1, K)))) # Mean

  V_beta_prior <- lapply(beta_dim, \(d) torch_eye(d) * 1) # Covariance matrix

  V_beta <- lapply(beta_dim, \(d) torch_eye(d) * 1) # Covariance matrix

  beta_trace <- lapply(beta_dim, \(d) torch_zeros(max_iter, d)) # Trace of beta

  # tau

  omega_prior <- torch_ones(L, L, dtype = torch_float()) # Dirichlet priors

  omega <- torch_ones(L, L, dtype = torch_float())

  omega_trace <- torch_zeros(max_iter, L, L) # Trace of omega

  # pi

  alpha_prior <- torch_ones(L, dtype = torch_float()) # Dirichlet prior

  alpha <- torch_ones(L, dtype = torch_float())

  alpha_trace <- torch_zeros(max_iter, L) # Trace of alpha

  # Z

  E_Z <- torch_ones(I, indT, L) / L

  E_Z_inter <- torch_ones(I, indT - 1, L, L) / (L * L)

  E_Z_trace <- torch_zeros(max_iter, I, indT - 1, L) # Trace of Z

  # xi

  xi <- update_xi(
    M_beta = M_beta,
    V_beta = V_beta,
    Delta_matrices = Delta_matrices
  )

  xi_trace <- torch_zeros(max_iter, J, L) # Trace of xi

  xi_trace[1, , ] <- xi

  if (elbo) {
    current_elbo <- compute_elbo(
      Y = Y,
      Delta_matrices = Delta_matrices,
      M_beta = M_beta,
      V_beta = V_beta,
      M_beta_prior = M_beta_prior,
      V_beta_prior = V_beta_prior,
      alpha = alpha,
      alpha_prior = alpha_prior,
      omega = omega,
      omega_prior = omega_prior,
      E_Z = E_Z,
      E_Z_inter = E_Z_inter,
      xi = xi
    )
    elbo_trace <- c(current_elbo)
  }

  # --------------------- Main loop --------------------- #

  cli_progress_bar("Running CAVI...", total = max_iter, type = "tasks")

  for (iter_ in seq(max_iter)) {
    # update Z

    log_phi <- E_log_phi(Y,
      M_beta = M_beta,
      V_beta = V_beta,
      xi = xi,
      Delta_matrices = Delta_matrices
    )

    log_kappa <- E_log_pi(alpha)

    log_eta <- E_log_omega(omega)

    update_Z(log_phi, log_kappa, log_eta) %->% c(E_Z, E_Z_inter)

    # update beta

    update_beta(
      Y = Y,
      K = K,
      M_beta_prior = M_beta_prior,
      V_beta_prior = V_beta_prior,
      Delta_matrices = Delta_matrices,
      E_Z = E_Z,
      xi = xi
    ) %->% c(M_beta, V_beta)

    for (j in seq(J)) {
      beta_trace[[j]][iter_, ] <- M_beta[[j]] # track the trace of beta
    }

    # update xi

    update_xi(
      M_beta = M_beta,
      V_beta = V_beta,
      Delta_matrices = Delta_matrices
    ) %->% xi

    xi_trace[iter_, , ] <- xi # track the trace of xi

    # update omega

    update_omega(
      omega_prior = omega_prior,
      E_Z_inter = E_Z_inter
    ) %->% omega

    # track the trace of omega

    omega_trace[iter_, , ] <- omega

    # update alpha
    update_alpha(
      alpha_prior = alpha_prior,
      E_Z = E_Z
    ) %->% alpha

    # track the trace of alpha

    alpha_trace[iter_, ] <- alpha

    if (elbo) {
      current_elbo <- compute_elbo(
        Y = Y,
        Delta_matrices = Delta_matrices,
        M_beta = M_beta,
        V_beta = V_beta,
        M_beta_prior = M_beta_prior,
        V_beta_prior = V_beta_prior,
        alpha = alpha,
        alpha_prior = alpha_prior,
        omega = omega,
        omega_prior = omega_prior,
        E_Z = E_Z,
        E_Z_inter = E_Z_inter,
        xi = xi
      )

      elbo_trace <- c(elbo_trace, current_elbo)
    }
    cli_progress_update()
  }

  cli_progress_done()

  end_time <- Sys.time()

  # --------------------- Post-hoc analysis --------------------- #

  if (is.null(data$ground_truth)) {
    res <- list(
      "beta" = sapply(M_beta, \(x) as_array(x)) |> t(),
      "beta_sd" = lapply(V_beta, \(v) torch_sqrt(v)),
      "profiles_index_hat" = E_Z$argmax(dim = 3),
      "pii" = as_array(alpha / alpha$sum()),
      "tau" = as_array(omega / omega$sum(2)$unsqueeze(2)),
      "alpha_trace" = as_array(alpha_trace),
      "omega_trace" = as_array(omega_trace),
      "beta_trace" = torch_stack(beta_trace) |> torch_transpose(1, 2) |> as_array(),
      "Q_hat" = Q_recovery(
        M_beta = M_beta,
        V_beta = V_beta,
        beta_hat_trace = beta_trace,
        E_Z = E_Z,
        alpha_level = alpha_level,
        Q_true = NULL
      ),
      "runtime" = as.numeric(difftime(end_time, start_time, units = "secs"))
    )
  } else {
    post_hoc <- Q_recovery(
      M_beta = M_beta,
      V_beta = V_beta,
      beta_hat_trace = beta_trace,
      E_Z = E_Z,
      alpha_level = alpha_level,
      Q_true = data$ground_truth$Q_matrix
    )

    # attribute accuracy

    profile_acc <- colMeans(post_hoc$profiles_index == data$ground_truth$profiles_index)

    # beta recovery

    M_beta_true <- array(0, dim = c(J, K + 1))

    for (j in seq(J)) {
      M_beta_true[j, c(TRUE, data$ground_truth$Q_matrix[j, ] == 1)] <- data$ground_truth$beta[[j]]
    }

    # compute the re-ordering mapping

    ord_map <- sapply(seq(L), \(l) sum(intToBin(l - 1, K) * 2^(post_hoc$ord - 1)) + 1)

    res <- list(
      "beta" = post_hoc$beta_hat,
      "beta_sd" = post_hoc$beta_hat_sd,
      "beta_true" = M_beta_true,
      "beta_trace" = post_hoc$beta_hat_trace,
      "Q_hat" = post_hoc$Q_hat,
      "Q_acc" = post_hoc$acc,
      "profiles_index_hat" = post_hoc$profiles_index,
      "profiles_acc" = profile_acc,
      "pii" = as_array(alpha / alpha$sum())[ord_map],
      "tau" = as_array(omega / omega$sum(2)$unsqueeze(2))[ord_map, ord_map],
      "alpha_trace" = as_array(alpha_trace)[, ord_map],
      "omega_trace" = as_array(omega_trace)[, ord_map, ord_map],
      "runtime" = as.numeric(difftime(end_time, start_time, units = "secs"))
    )
  }
  if (elbo) {
    res$"elbo" <- elbo_trace
  }
  return(res)
}

res <- HMLCDM_VB(data = data,
                      max_iter = 100,
                      alpha_level = 0.01,
                      elbo = TRUE)

# data1 <- data
# data1$ground_truth <- NULL
# 
# res1 <- HMLCDM_VB(
#   data = data1,
#   max_iter = 100,
#   alpha_level = 0.01,
#   elbo = TRUE
# )
