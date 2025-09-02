# Variational inference for the HMLCDM model

hmlcdm_vb_truncate <- function(data, max_iter = 100, alpha_level = 0.05,
                               elbo = FALSE, device = "auto") {
  # Setup device
  device <- get_device(device) # nolint

  # Print the number of threads Torch is using
  if (device == "cpu") {
    print(paste("Torch is using", torch_get_num_threads(), "threads")) # nolint
  }

  # Setup a timer
  start_time <- Sys.time()

  # --------------------- Retrieve and format data --------------------- #

  # Response matrix y, shape (i, t, j)

  y <- torch_tensor(data$y, device = device) # nolint

  # Number of observations, number of time points, number of items

  c(i, t, j) %<-% y$shape # nolint

  # Number of attributes
  k <- data$k

  # Number of latent profiles
  l <- 2^k

  # Generate the delta matrix for each item
  delta_mat <- lapply(seq(j), \(j) { # nolint
    build_delta( # nolint
      q = rep(1, k),
      interact = FALSE
    ) |>
      torch_tensor(device = device) # nolint
  })

  beta_dim <- rep(k + 1, j)

  # --------------------- Parameter initialization --------------------- #

  # Initialize beta

  m_beta_prior <- lapply(beta_dim, \(d) {
    torch_zeros(d, device = device) # nolint
  })

  m_beta <- lapply( # nolint
    beta_dim,
    \(d) {
      # Add small random perturbation to avoid identical initializations
      base_init <- c(-3, rep(2, k))
      noise <- torch_randn(d, device = device) * 0.1 # Small random noise # nolint
      torch_tensor(base_init, device = device) + noise # nolint
    }
  )

  v_beta_prior <- lapply(beta_dim, \(d) {
    torch_eye(d, device = device) # nolint
  })

  v_beta <- lapply(beta_dim, \(d) {
    torch_eye(d, device = device) # nolint
  })

  beta_trace <- lapply(
    beta_dim,
    \(d) {
      torch_zeros(max_iter, d, device = device) # nolint
    }
  )

  print("hi")

  # Initialize omega

  omega_prior <- torch_ones(l, l, dtype = torch_float(), device = device) # nolint

  omega <- torch_ones(l, l, dtype = torch_float(), device = device) + torch_randn(l, l, device = device) * 0.1 # nolint

  omega_trace <- torch_zeros(max_iter, l, l, device = device) # nolint

  # Initialize alpha

  alpha_prior <- torch_ones(l, dtype = torch_float(), device = device) # nolint

  alpha <- torch_ones(l, dtype = torch_float(), device = device) + torch_randn(l, device = device) * 0.1 # nolint

  alpha_trace <- torch_zeros(max_iter, l, device = device) # nolint

  # Initialize z

  # Initialize with random perturbation around uniform distribution
  e_z <- torch_ones(i, t, l, device = device) / l + torch_rand(i, t, l, device = device) * 0.01 # nolint
  # Ensure probabilities are positive and sum to 1 across latent classes
  e_z <- torch_abs(e_z) # nolint
  e_z <- e_z / e_z$sum(dim = 3, keepdim = TRUE) # nolint

  # Initialize transition expectations with random perturbation
  e_zz <- torch_ones(i, t - 1, l, l, device = device) / (l * l) + torch_rand(i, t - 1, l, l, device = device) * 0.001 # nolint
  # Ensure probabilities are positive and sum to 1 across destination classes
  e_zz <- torch_abs(e_zz) # nolint
  e_zz <- e_zz / e_zz$sum(dim = 4, keepdim = TRUE) # nolint

  e_z_trace <- torch_zeros(max_iter, i, t - 1, l, device = device) # nolint

  # Initialize xi

  xi <- update_xi_truncate(m_beta = m_beta, v_beta = v_beta, delta_mat = delta_mat) # nolint

  xi_trace <- torch_zeros(max_iter, j, l, device = device) # nolint

  xi_trace[1, , ] <- xi

  if (elbo) {
    current_elbo <- compute_elbo( # nolint
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

  cli_progress_bar("Running CAVI...", total = max_iter, type = "tasks") # nolint

  for (iter_ in seq(max_iter)) {
    # Update z
    log_phi <- e_log_phi( # nolint
      y = y,
      m_beta = m_beta,
      v_beta = v_beta,
      xi = xi,
      delta_mat = delta_mat
    )

    log_kappa <- e_log_pi(alpha) # nolint

    log_eta <- e_log_omega(omega) # nolint

    update_z(log_phi, log_kappa, log_eta) %->% c(e_z, e_zz) # nolint

    # Update beta
    update_beta( # nolint
      y = y,
      k = k,
      m_beta_prior = m_beta_prior,
      v_beta_prior = v_beta_prior,
      delta_mat = delta_mat,
      e_z = e_z,
      xi = xi
    ) %->% c(m_beta, v_beta) # nolint

    # Track the trace of beta
    for (j_iter in seq(j)) {
      beta_trace[[j_iter]][iter_, ] <- m_beta[[j_iter]]
    }

    # Update xi
    update_xi_truncate( # nolint
      m_beta = m_beta,
      v_beta = v_beta,
      delta_mat = delta_mat
    ) %->% xi # nolint

    # Track the trace of xi
    xi_trace[iter_, , ] <- xi

    # Update omega
    update_omega( # nolint
      omega_prior = omega_prior,
      e_zz = e_zz
    ) %->% omega # nolint

    # Track the trace of omega
    omega_trace[iter_, , ] <- omega

    # Update alpha
    update_alpha( # nolint
      alpha_prior = alpha_prior,
      e_z = e_z
    ) %->% alpha # nolint

    # Track the trace of alpha
    alpha_trace[iter_, ] <- alpha

    if (elbo) {
      current_elbo <- compute_elbo( # nolint
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
    cli_progress_update() # nolint
  }
  cli_progress_done() # nolint

  end_time <- Sys.time()
  print(m_beta)

  return(list('m_beta' = m_beta,
              'v_beta' = v_beta))

  # --------------------- Post-hoc analysis --------------------- #

  if (is.null(data$ground_truth)) {
    print("No ground truth available for evaluation.")
    res <- list(
      "beta" = sapply(m_beta, \(x) as_array(x)) |> t(), # nolint
      "beta_sd" = lapply(v_beta, \(v) torch_sqrt(v)), # nolint
      "profiles_index_hat" = e_z$argmax(dim = 3),
      "pii" = as_array(alpha / alpha$sum()),
      "tau" = as_array(omega / omega$sum(2)$unsqueeze(2)),
      "alpha_trace" = as_array(alpha_trace),
      "omega_trace" = as_array(omega_trace),
      "beta_trace" = as_array(beta_trace |>
        torch_stack() |> # nolint
        torch_transpose(1, 2)), # nolint
      # "q_mat_hat" = q_mat_recovery( # nolint
      #   m_beta = m_beta,
      #   v_beta = v_beta,
      #   beta_hat_trace = beta_trace,
      #   alpha_level = alpha_level,
      #   q_mat_true = NULL
      # ),
      "runtime" = as.numeric(difftime(end_time, start_time, units = "secs"))
    )
  } else {
    post_hoc <- q_mat_recovery( # nolint
      m_beta = m_beta,
      v_beta = v_beta,
      beta_hat_trace = beta_trace,
      alpha_level = alpha_level,
      q_mat_true = data$ground_truth$q_mat
    )

    # Recovery of beta
    m_beta_true <- array(0, dim = c(j, k + 1)) # nolint

    for (j_iter in seq(j)) {
      m_beta_true[j_iter, c(TRUE, data$ground_truth$q_mat[j_iter, ] == 1)] <-
        data$ground_truth$beta[[j_iter]]
    }

    # Compute the re-ordering mapping
    ord_map <- sapply(seq(l), \(l_iter) {
      binary_old <- int_to_bin(l_iter - 1, k)
      binary_new <- binary_old[post_hoc$ord]
      sum(binary_new * 2^(0:(k - 1))) + 1
    })

    # attribute accuracy

    profiles_index <- matrix(ord_map[as_array(e_z$argmax(dim = 3))], nrow = i) # nolint

    profile_acc <- colMeans(profiles_index == data$ground_truth$profiles_index) # nolint

    res <- list(
      "beta" = post_hoc$beta_hat,
      "beta_sd" = post_hoc$beta_hat_sd,
      "beta_true" = m_beta_true,
      "beta_trace" = post_hoc$beta_hat_trace,
      "Q_hat" = post_hoc$Q_hat,
      "Q_acc" = post_hoc$acc,
      "profiles_index_hat" = profiles_index,
      "profiles_acc" = profile_acc,
      "pii" = as_array(alpha / alpha$sum())[ord_map], # nolint
      "tau" = as_array(omega / omega$sum(2)$unsqueeze(2))[ord_map, ord_map],
      "alpha_trace" = as_array(alpha_trace)[, ord_map],
      "omega_trace" = as_array(omega_trace)[, ord_map, ord_map],
      "runtime" = as.numeric(difftime(end_time, start_time, units = "secs"))
    )
  }
  if (elbo) {
    res$"elbo" <- elbo_trace
  }
  res
}
