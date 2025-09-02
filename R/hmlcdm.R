# Variational inference for the HMLCDM model

hmlcdm_vb <- function(data, max_iter = 100, alpha_level = 0.05,
                      elbo = FALSE, device = "auto") {
  # Setup device
  device <- get_device(device)

  # Print the number of threads Torch is using
  if (device == "cpu") {
    print(paste("Torch is using", torch_get_num_threads(), "threads"))
  }

  # Setup a timer
  start_time <- Sys.time()

  # --------------------- Retrieve and format data --------------------- #

  # Response matrix y, shape (i, t, j)

  y <- torch_tensor(data$y, device = device)

  # Number of observations, number of time points, number of items

  c(i, t, j) %<-% y$shape

  # Number of attributes
  k <- data$k

  # Number of latent profiles
  l <- 2^k

  # Generate the delta matrix for each item
  delta_mat <- lapply(seq(j), \(j) {
    build_delta(
      q = rep(1, k),
      interact = FALSE
    ) |>
      torch_tensor(device = device)
  })

  beta_dim <- rep(k + 1, j)

  # --------------------- Parameter initialization --------------------- #

  # Initialize beta

  m_beta_prior <- lapply(beta_dim, \(d) {
    torch_zeros(d, device = device)
  })

  m_beta <- lapply(
    beta_dim,
    \(d) {
      # Add small random perturbation to avoid identical initializations
      base_init <- c(-3, rep(2, k))
      noise <- torch_randn(d, device = device) * 0.001 # Small random noise
      torch_tensor(base_init, device = device) + noise
    }
  )

  v_beta_prior <- lapply(beta_dim, \(d) {
    torch_eye(d, device = device) * 1
  })

  v_beta <- lapply(beta_dim, \(d) {
    torch_eye(d, device = device) * 1
  })

  beta_trace <- lapply(
    beta_dim,
    \(d) {
      torch_zeros(max_iter, d, device = device)
    }
  )

  # Initialize omega

  omega_prior <- torch_ones(l, l, dtype = torch_float(), device = device)

  omega <- torch_ones(l, l, dtype = torch_float(), device = device) # + torch_randn(l, l, device = device) * 0.1

  omega_trace <- torch_zeros(max_iter, l, l, device = device)

  # Initialize alpha

  alpha_prior <- torch_ones(l, dtype = torch_float(), device = device)

  alpha <- torch_ones(l, dtype = torch_float(), device = device) # + torch_randn(l, device = device) * 0.1

  alpha_trace <- torch_zeros(max_iter, l, device = device)

  # Initialize z

  # Initialize with random perturbation around uniform distribution
  e_z <- torch_ones(i, t, l, device = device) / l # + torch_rand(i, t, l, device = device) * 0.01
  # Ensure probabilities are positive and sum to 1 across latent classes
  e_z <- torch_abs(e_z)
  e_z <- e_z / e_z$sum(dim = 3, keepdim = TRUE)

  # Initialize transition expectations with random perturbation
  e_zz <- torch_ones(i, t - 1, l, l, device = device) / (l * l) + torch_rand(i, t - 1, l, l, device = device) * 0.001
  # Ensure probabilities are positive and sum to 1 across destination classes
  e_zz <- torch_abs(e_zz)
  e_zz <- e_zz / e_zz$sum(dim = 4, keepdim = TRUE)

  e_z_trace <- torch_zeros(max_iter, i, t - 1, l, device = device)

  # Initialize xi

  xi <- update_xi(m_beta = m_beta, v_beta = v_beta, delta_mat = delta_mat)

  xi_trace <- torch_zeros(max_iter, j, l, device = device)

  xi_trace[1, , ] <- xi

  if (elbo) {
    current_elbo <- compute_elbo(
      y = y,
      delta_mat = delta_mat,
      m_beta = m_beta,
      v_beta = v_beta,
      m_beta_prior = m_beta_prior,
      v_beta_prior = v_beta_prior,
      alpha = alpha,
      alpha_prior = alpha_prior,
      omega = omega,
      omega_prior = omega_prior,
      e_z = e_z,
      e_zz = e_zz,
      xi = xi
    )
    elbo_trace <- c(current_elbo)
  }

  # --------------------- Main loop --------------------- #

  cli_progress_bar("Running CAVI...", total = max_iter, type = "tasks")

  for (iter_ in seq(max_iter)) {
    # Update z
    log_phi <- e_log_phi(
      y = y,
      m_beta = m_beta,
      v_beta = v_beta,
      xi = xi,
      delta_mat = delta_mat
    )

    log_kappa <- e_log_pi(alpha)

    log_eta <- e_log_omega(omega)

    update_z(log_phi, log_kappa, log_eta) %->% c(e_z, e_zz)

    # Update beta
    update_beta(
      y = y,
      k = k,
      m_beta_prior = m_beta_prior,
      v_beta_prior = v_beta_prior,
      delta_mat = delta_mat,
      e_z = e_z,
      xi = xi
    ) %->% c(m_beta, v_beta)

    # Track the trace of beta
    for (j_iter in seq(j)) {
      beta_trace[[j_iter]][iter_, ] <- m_beta[[j_iter]]
    }

    # Update xi
    update_xi(
      m_beta = m_beta,
      v_beta = v_beta,
      delta_mat = delta_mat
    ) %->% xi

    # Track the trace of xi
    xi_trace[iter_, , ] <- xi

    # Update omega
    update_omega(
      omega_prior = omega_prior,
      e_zz = e_zz
    ) %->% omega

    # Track the trace of omega
    omega_trace[iter_, , ] <- omega

    # Update alpha
    update_alpha(
      alpha_prior = alpha_prior,
      e_z = e_z
    ) %->% alpha

    # Track the trace of alpha
    alpha_trace[iter_, ] <- alpha

    if (elbo) {
      current_elbo <- compute_elbo(
        y = y,
        delta_mat = delta_mat,
        m_beta = m_beta,
        v_beta = v_beta,
        m_beta_prior = m_beta_prior,
        v_beta_prior = v_beta_prior,
        alpha = alpha,
        alpha_prior = alpha_prior,
        omega = omega,
        omega_prior = omega_prior,
        e_z = e_z,
        e_zz = e_zz,
        xi = xi
      )
      elbo_trace <- c(elbo_trace, current_elbo)
    }
    cli_progress_update()
  }
  cli_progress_done()

  end_time <- Sys.time()

  res <- list(
    "beta" = sapply(m_beta, \(x) as_array(x)) |> t(),
    "beta_cov" = lapply(v_beta, \(v) as_array(v)),
    "pii" = as_array(alpha / alpha$sum()),
    "tau" = as_array(omega / omega$sum(2)$unsqueeze(2)),
    "alpha_trace" = as_array(alpha_trace),
    "omega_trace" = as_array(omega_trace),
    "beta_trace" = as_array(beta_trace |>
      torch_stack() |>
      torch_transpose(1, 2)),
    "profiles_index_hat" = as_array(e_z$argmax(dim = 3)),
    "runtime" = as.numeric(difftime(end_time, start_time, units = "secs"))
  )

  if (elbo) {
    res$"elbo" <- elbo_trace
  }

  res
}



# --------------------- Post-hoc analysis --------------------- #


post_hoc <- function(res, alpha_level = 0.05, q_mat_true = NULL) {
  res_post <- res

  c(j, k) %<-% dim(res$beta)

  k <- k - 1

  beta_hat <- res$beta

  # Compute standard deviations
  beta_hat_sd <- sapply(res$beta_cov, \(v) sqrt(diag(v))) |> t()

  beta_hat_trace <- res$beta_trace

  if (is.null(q_mat_true)) {
    q_mat_hat <- q_mat_recovery(
      beta_hat = beta_hat,
      beta_hat_sd = beta_hat_sd,
      beta_hat_trace = beta_hat_trace,
      alpha_level = alpha_level,
      q_mat_true = q_mat_true
    )
    res_post$beta_hat_sd <- beta_hat_sd
    res_post$q_mat_hat <- q_mat_hat
  } else {
    post_hoc <- q_mat_recovery(
      beta_hat = beta_hat,
      beta_hat_sd = beta_hat_sd,
      beta_hat_trace = beta_hat_trace,
      alpha_level = alpha_level,
      q_mat_true = data$ground_truth$q_mat
    )

    # Recovery of beta
    m_beta_true <- array(0, dim = c(j, k + 1))

    for (j_iter in seq(j)) {
      m_beta_true[j_iter, c(TRUE, data$ground_truth$q_mat[j_iter, ] == 1)] <-
        data$ground_truth$beta[[j_iter]]
    }

    res_post$beta_true <- m_beta_true

    # Compute the re-ordering mapping
    ord_map <- sapply(seq(2^k), \(l_iter) {
      binary_old <- int_to_bin(l_iter - 1, k)
      binary_new <- binary_old[post_hoc$ord]
      sum(binary_new * 2^(0:(k - 1))) + 1
    })

    # Compute inverse mapping: inv_ord_map[j] = which old profile should be in position j
    inv_ord_map <- integer(2^k)
    inv_ord_map[ord_map] <- seq(2^k)

    res_post$profiles_index_hat <- matrix(ord_map[res$profiles_index_hat], nrow = nrow(res$profiles_index_hat))

    res_post$profiles_acc <- colMeans(res_post$profiles_index_hat == data$ground_truth$profiles_index)

    res_post$q_mat_hat <- post_hoc$Q_hat

    res_post$q_mat_acc <- post_hoc$acc

    res_post$pii <- res$pii[inv_ord_map]

    res_post$tau <- res$tau[inv_ord_map, inv_ord_map]

    res_post$alpha_trace <- res$alpha_trace[, inv_ord_map]

    res_post$omega_trace <- res$omega_trace[, inv_ord_map, inv_ord_map]

    res_post$beta <- post_hoc$beta_hat

    res_post$beta_sd <- post_hoc$beta_hat_sd

    res_post$beta_trace <- post_hoc$beta_hat_trace
  }

  res_post
}
