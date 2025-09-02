# ------------------------------------------------------------------------- #

#' Jaakkola-Jordan Lower Bound Function
#'
#' Computes the Jaakkola-Jordan lower bound function for variational inference
#' with logistic regression. This function approximates the logistic function
#' to enable tractable variational updates.
#'
#' @param xi A torch tensor containing the variational parameters
#' @param epsilon A small positive value to avoid numerical instability
#'   (default: 1e-10)
#'
#' @return A torch tensor with the same shape as xi containing the computed
#'   Jaakkola-Jordan function values: tanh(|xi|/2) / (4 * |xi|)
#'
#' @details The Jaakkola-Jordan bound enables the use of conjugate variational
#'   inference for logistic regression by providing a quadratic lower bound
#'   for the logistic function.
#'
#' @references
#' Jaakkola, T. S., & Jordan, M. I. (1997). A variational approach to Bayesian
#' logistic regression models and their extensions.
#'
#' @examples
#' \dontrun{
#' xi <- torch_randn(5, 3)
#' result <- jj_func(xi)
#' }
#'
#' @importFrom torch torch_abs torch_tanh
#' @export
jj_func <- function(xi, epsilon = 1e-10) {
  x <- torch_abs(xi) + epsilon
  return(torch_tanh(x / 2) / (4 * x))
}

# ------------------------------------------------------------------------- #

#' Update Beta Parameters for Variational Inference
#'
#' Updates the variational posterior parameters for item-effect parameters
#' (beta) in the Hidden Markov Log-Linear Cognitive Diagnostic Model using
#' coordinate ascent variational inference.
#'
#' @param y A torch tensor of shape (i, t, j) containing binary response data,
#'   where i is the number of individuals, t is the number of time points,
#'   and j is the number of items
#' @param k An integer specifying the number of attributes
#' @param m_beta_prior A list of length j containing prior mean vectors for
#'   beta parameters
#' @param v_beta_prior A list of length j containing prior covariance matrices
#'   for beta parameters
#' @param delta_mat A list of length j containing design matrices for each item
#' @param z A torch tensor of shape (i, t, l) containing expected latent class
#'   memberships, where l = 2^k is the number of latent classes
#' @param xi A torch tensor of shape (j, l) containing auxiliary variational
#'   parameters for the Jaakkola-Jordan bound
#'
#' @return A list containing:
#'   \itemize{
#'     \item m_beta: List of length j with updated posterior mean vectors
#'     \item v_beta: List of length j with updated posterior covariance matrices
#'   }
#'
#' @details This function implements the coordinate ascent variational inference
#'   update for the item-effect parameters in the HM-LCDM. The updates use the
#'   Jaakkola-Jordan lower bound to handle the nonconjugate logistic likelihood.
#'
#' @examples
#' \dontrun{
#' # Assuming appropriate torch tensors and parameters are available
#' result <- update_beta(y, k, m_beta_prior, v_beta_prior, delta_mat, e_z, xi)
#' updated_means <- result$m_beta
#' updated_covariances <- result$v_beta
#' }
#'
#' @export
update_beta <- function(y, k, m_beta_prior, v_beta_prior, delta_mat, e_z, xi) {
  # Extract dimensions from response tensor
  c(i, t, j) %<-% y$shape

  # Sum over individuals and time points to get class weights
  weights_z <- e_z$sum(c(1, 2))

  # Initialize storage for posterior parameters
  m_beta <- vector("list", j)

  v_beta <- vector("list", j)

  # Update parameters for each item
  for (j_iter in seq(j)) {
    delta_mat_j <- delta_mat[[j_iter]]

    # Update posterior covariance matrix
    v_beta[[j_iter]] <- torch_inverse(
      v_beta_prior[[j_iter]]$inverse() +
        2 * (
          delta_mat_j$t() %@%
            torch_diag_embed(weights_z * jj_func(xi[j_iter, ])) %@%
            delta_mat_j
        )
    )

    # Update posterior mean vector
    m_beta[[j_iter]] <- v_beta[[j_iter]] %@% (
      v_beta_prior[[j_iter]]$inverse() %@% m_beta_prior[[j_iter]] +
        delta_mat_j$t() %@% torch_einsum(
          "ntl,nt->l", list(e_z, y[, , j_iter] - 1 / 2)
        )
    )
  }

  list(
    "m_beta" = m_beta,
    "v_beta" = v_beta
  )
}

# ------------------------------------------------------------------------- #

#' Update Auxiliary Variables Xi for Item-Response
#'
#' Updates the auxiliary variational parameters (xi) for the Jaakkola-Jordan
#' lower bound in variational inference for logistic regression. The parameters
#' are used to approximate the logistic function in the item-response model.
#'
#' @param m_beta A list of length j containing posterior mean vectors for
#'   item-effect parameters (beta), where each element is a torch tensor
#' @param v_beta A list of length j containing posterior covariance matrices
#'   for item-effect parameters (beta), where each element is a torch tensor
#' @param delta_mat A list of length j containing design matrices for each item,
#'   where each element is a torch tensor of shape (l, k+1) with l = 2^k
#'
#' @return A torch tensor of shape (j, l) containing the updated auxiliary
#'   variables xi, where j is the number of items and l = 2^k is the number
#'   of latent classes
#'
#' @details This function computes the optimal auxiliary variables for the
#'   Jaakkola-Jordan bound. The auxiliary variables are used to
#'   create a quadratic lower bound for the logistic function.
#'
#' @examples
#' \dontrun{
#' # Assuming appropriate torch tensors and parameters are available
#' xi_updated <- update_xi(m_beta, v_beta, delta_mat)
#' }
#'
#' @export
update_xi <- function(m_beta, v_beta, delta_mat) {
  j <- length(m_beta)

  # Number of attributes
  k <- m_beta[[1]]$shape - 1

  # Use same device as input tensors
  device <- m_beta[[1]]$device

  xi <- torch_zeros(j, 2^k, device = device)

  for (j_iter in seq(j)) {
    e_beta2 <- m_beta[[j_iter]]$outer(m_beta[[j_iter]]) + v_beta[[j_iter]]

    xi[j_iter, ] <- delta_mat[[j_iter]] %@% e_beta2 %@% delta_mat[[j_iter]]$t() |>
      torch_diag() |>
      torch_sqrt()
  }

  xi
}

#' @importFrom tmvtnorm mtmvnorm
#' @importFrom Matrix nearPD
#' @export
update_xi_truncate <- function(m_beta, v_beta, delta_mat) {
  j <- length(m_beta)

  # Number of attributes
  k <- m_beta[[1]]$shape - 1

  # Use same device as input tensors
  device <- m_beta[[1]]$device

  xi <- torch_zeros(j, 2^k, device = device)

  lower <- c(-Inf, rep(0, k))
  upper <- rep(5, k + 1)

  for (j_iter in seq(j)) {
    tmp <- mtmvnorm(
      mean = as_array(m_beta[[j_iter]]), sigma = as.matrix(nearPD(as_array(v_beta[[j_iter]]), corr = TRUE)$mat),
      lower = lower, upper = upper
    )

    mean_beta <- torch_tensor(tmp$tmean, device = device)

    cov_beta <- torch_tensor(as.matrix(nearPD(tmp$tvar, corr = TRUE)$mat), device = device)

    e_beta2 <- mean_beta$outer(mean_beta) + cov_beta

    xi[j_iter, ] <- delta_mat[[j_iter]] %@% e_beta2 %@% delta_mat[[j_iter]]$t() |>
      torch_diag() |>
      torch_sqrt()
  }

  xi
}

# ------------------------------------------------------------------------- #

#' Update Transition Probabilities (Omega)
#'
#' Updates the variational posterior parameters for the transition probabilities
#' (omega) in the Hidden Markov Log-Linear Cognitive Diagnostic Model using
#' coordinate ascent variational inference.
#'
#' @param omega_prior A torch tensor of shape (l, l) containing the prior
#'   parameters for the transition probability matrix, where l = 2^k is the
#'   number of latent classes
#' @param e_zz A torch tensor of shape (i, t-1, l, l) containing the expected
#'   sufficient statistics for the transition counts, where i is the number
#'   of individuals and t is the number of time points
#'
#' @return A torch tensor of shape (l, l) containing the updated posterior
#'   parameters for the transition probability matrix
#'
#' @details This function implements the conjugate update for the Dirichlet
#'   posterior of the transition probabilities by adding the expected transition
#'   counts (summed over individuals and time points) to the prior parameters.
#'
#' @examples
#' \dontrun{
#' # Assuming appropriate torch tensors are available
#' omega_updated <- update_omega(omega_prior, e_zz)
#' }
#'
#' @export
update_omega <- function(omega_prior, e_zz) {
  omega_prior + e_zz$sum(c(1, 2))
}

# ------------------------------------------------------------------------- #

#' Update Initial Distribution of Latent Attributes (Alpha)
#'
#' Updates the variational posterior parameters for the initial distribution
#' (alpha) of latent attributes in the Hidden Markov Log-Linear Cognitive
#' Diagnostic Model using coordinate ascent variational inference.
#'
#' @param alpha_prior A torch tensor of shape (l,) containing the prior
#'   parameters for the initial distribution, where l = 2^k is the number
#'   of latent classes
#' @param e_z A torch tensor of shape (i, t, l) containing the expected
#'   latent class memberships, where i is the number of individuals,
#'   t is the number of time points, and l is the number of latent classes
#'
#' @return A torch tensor of shape (l,) containing the updated posterior
#'   parameters for the initial distribution
#'
#' @details This function implements the conjugate update for the Dirichlet
#'   posterior of the initial latent class distribution by adding the expected
#'   initial class counts (summed over individuals at the first time point)
#'   to the prior parameters.
#'
#' @examples
#' \dontrun{
#' # Assuming appropriate torch tensors are available
#' alpha_updated <- update_alpha(alpha_prior, e_z)
#' }
#'
#' @export
update_alpha <- function(alpha_prior, e_z) {
  alpha_prior + e_z[, 1, ]$sum(1)
}

# ------------------------------------------------------------------------- #

#' Update Latent Attribute Expectations (Z)
#'
#' Computes the expected latent class memberships and transition counts
#' using the forward-backward algorithm for the Hidden Markov Log-Linear
#' Cognitive Diagnostic Model.
#'
#' @param log_phi A torch tensor of shape (i, t, l) containing the log item
#'   response probabilities for each individual, time point, and latent class
#' @param log_kappa A torch tensor of shape (l,) containing the log initial
#'   distribution parameters for latent classes
#' @param log_eta A torch tensor of shape (l, l) containing the log transition
#'   probability matrix between latent classes
#' @param epsilon A small positive value to avoid numerical instability
#'   (default: 1e-10)
#'
#' @return A list containing:
#'   \itemize{
#'     \item First element: A torch tensor of shape (i, t, l) with expected
#'       latent class memberships
#'     \item Second element: A torch tensor of shape (i, t-1, l, l) with
#'       expected transition counts between consecutive time points
#'   }
#'
#' @details This function implements the forward-backward algorithm to compute
#'   the posterior expectations of latent class memberships and transitions.
#'   The algorithm uses log-space computations to maintain numerical stability.
#'
#' @examples
#' \dontrun{
#' # Assuming appropriate torch tensors are available
#' result <- update_z(log_phi, log_kappa, log_eta)
#' e_z <- result[[1]]
#' e_zz <- result[[2]]
#' }
#'
#' @importFrom torch torch_zeros
#' @export
update_z <- function(log_phi, log_kappa, log_eta, epsilon = 1e-10) {
  c(i, t, l) %<-% log_phi$shape

  # Use same device as input tensors
  device <- log_phi$device

  log_f <- torch_zeros(c(i, t, l), dtype = torch_float(), device = device)

  log_b <- torch_zeros(c(i, t, l), dtype = torch_float(), device = device)

  log_f[, 1, ] <- log_kappa + log_phi[, 1, ]

  for (t_iter in 2:t) {
    log_f[, t_iter, ] <- log_phi[, t_iter, ] +
      torch_logsumexp(log_f[, t_iter - 1, , NULL] + log_eta, 2)

    log_b[, t - t_iter + 1, ] <- torch_logsumexp(
      log_phi[, t - t_iter + 2, , NULL] +
        log_b[, t - t_iter + 2, , NULL] + log_eta, 2
    )
  }

  # add epsilon to avoid numerical issues
  e_z <- nnf_softmax(log_f + log_b, 3) + epsilon

  e_z <- e_z / e_z$sum(dim = 3, keepdim = TRUE) # renormalise

  temp <- log_f[, 1:(t - 1), , NULL] + log_b[, 2:t, NULL, ] +
    log_eta + log_phi[, 2:t, NULL, ]

  # add epsilon to avoid numerical issues
  e_zz <- nnf_softmax(temp$view(c(i, t - 1, -1)), 3) + epsilon

  e_zz <- e_zz / e_zz$sum(dim = 3, keepdim = TRUE) # renormalise

  e_zz <- e_zz$view(c(i, t - 1, l, l))

  list(e_z, e_zz)
}

# ------------------------------------------------------------------------- #

#' Compute Expectation of Log Initial Distribution
#'
#' Computes the expectation of the logarithm of the initial distribution
#' parameters (pi) under a Dirichlet posterior distribution.
#'
#' @param alpha A torch tensor of shape (l,) containing the posterior
#'   Dirichlet parameters for the initial distribution, where l = 2^k
#'   is the number of latent classes
#'
#' @return A torch tensor of shape (l,) containing the expected values
#'   of log(pi_l) for each latent class l
#'
#' @details This function computes the expectation of log(pi_l) under a
#'   Dirichlet posterior distribution. This expectation is used in
#'   the variational lower bound computation.
#'
#' @examples
#' \dontrun{
#' alpha <- torch_tensor(c(2.5, 3.1, 4.2))
#' e_log_pi <- e_log_pi(alpha)
#' }
#'
#' @importFrom torch torch_digamma
#' @export
e_log_pi <- function(alpha) {
  torch_digamma(alpha) - torch_digamma(alpha$sum())
}

# ------------------------------------------------------------------------- #

#' Compute Expectation of Log Transition Matrix
#'
#' Computes the expectation of the logarithm of the transition matrix
#' parameters (omega) under a Dirichlet posterior distribution.
#'
#' @param omega A torch tensor of shape (l, l) containing the posterior
#'   Dirichlet parameters for the transition matrix, where l = 2^k is
#'   the number of latent classes
#'
#' @return A torch tensor of shape (l, l) containing the expected values
#'   of log(ω_{jk}) for transitions from latent class j to class k
#'
#' @details This function computes E[log(ω_{jk})] = ψ(ω_{jk}) - ψ(Σ_k ω_{jk})
#'   where ψ is the digamma function. This expectation is used in the
#'   variational lower bound computation for the transition probabilities.
#'
#' @examples
#' \dontrun{
#' omega <- torch_tensor(matrix(c(3.2, 1.8, 2.1, 4.5), nrow = 2))
#' e_log_omega <- e_log_omega(omega)
#' }
#'
#' @importFrom torch torch_digamma
#' @export
e_log_omega <- function(omega) {
  torch_digamma(omega) - torch_digamma(omega$sum(1))
}

# ------------------------------------------------------------------------- #

#' Compute Expected Log Item Response Probabilities
#'
#' Computes the expected log item response probabilities (log phi) for each
#' individual, time point, and latent class in the Hidden Markov Log-Linear
#' Cognitive Diagnostic Model using variational posterior parameters.
#'
#' @param Y A torch tensor of shape (i, t, j) containing binary response data,
#'   where i is the number of individuals, t is the number of time points,
#'   and j is the number of items
#' @param M_beta A list of length j containing posterior mean vectors for
#'   item-effect parameters (beta)
#' @param V_beta A list of length j containing posterior covariance matrices
#'   for item-effect parameters (beta)
#' @param xi A torch tensor of shape (j, l) containing auxiliary variational
#'   parameters for the Jaakkola-Jordan bound, where l = 2^k is the number
#'   of latent classes
#' @param Delta_matrices A list of length j containing design matrices for
#'   each item, where each element is a torch tensor of shape (l, k+1)
#'
#' @return A torch tensor of shape (i, t, l) containing the expected log
#'   item response probabilities for each individual, time point, and
#'   latent class
#'
#' @details This function implements the computation of expected log item
#'   response probabilities using the Jaakkola-Jordan lower bound for the
#'   logistic function. The computation accounts for uncertainty in the
#'   item-effect parameters through their posterior distributions.
#'
#' @examples
#' \dontrun{
#' # Assuming appropriate torch tensors and parameters are available
#' log_phi <- E_log_phi(Y, M_beta, V_beta, xi, Delta_matrices)
#' }
#'
#' @export
e_log_phi <- function(y, m_beta, v_beta, xi, delta_mat) {
  j <- length(m_beta)
  k <- m_beta[[1]]$shape - 1

  # Use same device as input tensors
  device <- y$device
  f_beta <- torch_zeros(j, 2^k, device = device)
  f_beta2 <- torch_zeros(j, 2^k, device = device)

  for (j_iter in seq(j)) {
    f_beta[j_iter, ] <- delta_mat[[j_iter]] %@% m_beta[[j_iter]]

    e_beta2 <- m_beta[[j_iter]]$outer(m_beta[[j_iter]]) + v_beta[[j_iter]]

    f_beta2[j_iter, ] <- (delta_mat[[j_iter]] %@% e_beta2 %@%
      delta_mat[[j_iter]]$t())$diagonal()
  }

  log_phi <- (
    (y$unsqueeze(-1) - 1 / 2) * f_beta +
      (nnf_logsigmoid(xi) - xi / 2 - jj_func(xi) * (f_beta2 - xi$square()))
  )$sum(dim = 3)

  log_phi
}

# ------------------------------------------------------------------------- #

#' Q-Matrix Recovery via Statistical Testing
#'
#' Performs post-hoc Q-matrix recovery by testing the significance of
#' item-effect parameters using Benjamini-Hochberg multiple testing correction
#' and optional permutation search for optimal attribute ordering.
#'
#' @param m_beta A list of length j containing posterior mean vectors for
#'   item-effect parameters (beta), where each element is a torch tensor
#' @param v_beta A list of length j containing posterior covariance matrices
#'   for item-effect parameters (beta), where each element is a torch tensor
#' @param beta_hat_trace A tensor of shape (max_iter, j, k+1) containing the
#'   trace of beta parameter estimates across iterations
#' @param alpha_level Significance level for the Benjamini-Hochberg procedure
#'   (default: 0.05)
#' @param q_mat_true Optional true Q-matrix of shape (j, k) for evaluating
#'   recovery accuracy. If provided, permutation search is performed to
#'   find the best attribute ordering (default: NULL)
#'
#' @return If q_mat_true is NULL, returns a binary matrix Q_hat of shape (j, k).
#'   If q_mat_true is provided, returns a list containing:
#'   \itemize{
#'     \item q_mat_hat: Recovered Q-matrix with optimal attribute ordering
#'     \item acc: Recovery accuracy (proportion of correctly identified entries)
#'     \item beta_hat: Parameter estimates with optimal ordering
#'     \item beta_hat_sd: Parameter standard deviations with optimal ordering
#'     \item beta_hat_trace: Parameter traces with optimal ordering
#'     \item ord: Optimal attribute ordering indices
#'   }
#'
#' @details This function identifies which attributes are required for each item
#'   by testing if the corresponding item-effect parameters are significantly
#'   greater than zero. It uses the Benjamini-Hochberg procedure to control
#'   the false discovery rate. When a true Q-matrix is provided, it searches
#'   over all possible attribute permutations to find the ordering that
#'   maximizes recovery accuracy.
#'
#' @examples
#' \dontrun{
#' # Basic Q-matrix recovery
#' q_mat_recovered <- q_mat_recovery(m_beta, v_beta, beta_trace)
#'
#' # With true Q-matrix for evaluation
#' result <- q_mat_recovery(m_beta, v_beta, beta_trace, q_mat_true = true_Q)
#' best_q_mat <- result$q_mat_hat
#' accuracy <- result$acc
#' }
#'
#' @export
q_mat_recovery <- function(beta_hat, beta_hat_sd, beta_hat_trace,
                           alpha_level = 0.05, q_mat_true = NULL) {
  c(j, k) %<-% dim(beta_hat)

  k <- k - 1

  # Remove the intercept term if all values are negative
  intercept_detected <- FALSE

  for (k_iter in seq(k + 1)) {
    if (all(beta_hat[, k_iter] < 0)) {
      cat("Column", k_iter, "is the intercept, will be removed. \n")

      beta_hat_main <- beta_hat[, -k_iter]

      beta_hat_sd_main <- beta_hat_sd[, -k_iter]

      beta_hat_trace_main <- beta_hat_trace[, , -k_iter]

      intercept_detected <- TRUE

      beta_hat_intercept <- beta_hat[, k_iter]

      beta_hat_sd_intercept <- beta_hat_sd[, k_iter]

      beta_hat_trace_intercept <- beta_hat_trace[, , k_iter]

      break
    }
  }

  if (!intercept_detected) {
    stop("No intercept term detected.")
  }

  # perform one-side z-test for each entry
  z_score <- beta_hat_main / beta_hat_sd_main

  log_p_value <- pnorm(z_score, log.p = TRUE, lower.tail = FALSE) |> as.vector()

  m <- length(log_p_value)

  o <- order(log_p_value)

  log_p_value_sorted <- log_p_value[o]

  log_q_value <- log(seq(m) / m * alpha_level) # Benjamini-Hochberg procedure

  n <- max(which(log_p_value_sorted <= log_q_value), 0L)

  sig <- logical(m)

  if (n > 0) {
    sig[o[1:n]] <- TRUE
  }

  if (!any(sig)) {
    warning("No attribute passed the BH threshold at alpha = ", alpha_level, "
    .")

    q_mat_hat <- matrix(0L, j, k)
  } else {
    q_mat_hat <- matrix(sig,
      nrow = nrow(beta_hat_main),
      ncol = ncol(beta_hat_main)
    ) * 1L
  }

  if (!is.null(q_mat_true)) {
    q_mat_hat_best <- q_mat_hat

    idx_best <- NULL

    acc <- 0

    it <- iterpc(k, k, ordered = TRUE)

    repeat {
      if (acc == 1) {
        print("Perfect Q-Matrix recovery achieved.")
        break
      }

      idx <- getnext(it, d = 1, drop = TRUE)

      if (length(idx) == 0) {
        break
      } # iterator exhausted
      if (mean((q_mat_hat[, idx]) == q_mat_true) > acc) {
        acc <- mean((q_mat_hat[, idx]) == q_mat_true)

        q_mat_hat_best <- q_mat_hat[, idx]

        idx_best <- idx

        print(paste0(
          "Permutating ... Best Q-Matrix recovery accuracy is ",
          round(acc * 100, 3), "%"
        ))
      }
    }

    beta_hat_permuted <- cbind(beta_hat_intercept, beta_hat_main[, idx_best])

    colnames(beta_hat_permuted) <- NULL

    beta_hat_sd_permuted <- cbind(
      beta_hat_sd_intercept,
      beta_hat_sd_main[, idx_best]
    )

    colnames(beta_hat_sd_permuted) <- NULL

    beta_hat_trace_permuted <- array(dim = dim(beta_hat_trace))

    beta_hat_trace_permuted[, , 1] <- beta_hat_trace_intercept

    beta_hat_trace_permuted[, , 2:(k + 1)] <- beta_hat_trace_main[, , idx_best]

    q_mat_hat <- q_mat_hat_best

    return(list(
      "Q_hat" = q_mat_hat,
      "acc" = acc,
      "beta_hat" = beta_hat_permuted,
      "beta_hat_sd" = beta_hat_sd_permuted,
      "beta_hat_trace" = beta_hat_trace_permuted,
      "ord" = idx_best
    ))
  }
  q_mat_hat
}

# ------------------------------------------------------------------------- #

#' Compute Evidence Lower Bound (ELBO)
#'
#' Computes the Evidence Lower Bound (ELBO) for the Hidden Markov Log-Linear
#' Cognitive Diagnostic Model using variational inference. The ELBO serves as
#' a lower bound on the log marginal likelihood and is used to monitor
#' convergence and assess model fit.
#'
#' @param y A torch tensor of shape (i, t, j) containing binary response,
#'   data where i is the number of individuals, t is the number of time points,
#'   and j is the number of items
#' @param delta_mat A list of length j containing design matrices for
#'   each item, where each element is a torch tensor of shape (l, k+1)
#' @param m_beta A list of length j containing posterior mean vectors for
#'   item-effect parameters (beta)
#' @param v_beta A list of length j containing posterior covariance matrices
#'   for item-effect parameters (beta)
#' @param m_beta_prior A list of length j containing prior mean vectors for
#'   item-effect parameters (beta)
#' @param v_beta_prior A list of length j containing prior covariance matrices
#'   for item-effect parameters (beta)
#' @param alpha A torch tensor of shape (l,) containing posterior Dirichlet
#'   parameters for the initial distribution, where l = 2^k
#' @param alpha_prior A torch tensor of shape (l,) containing prior Dirichlet
#'   parameters for the initial distribution
#' @param omega A torch tensor of shape (l, l) containing posterior Dirichlet
#'   parameters for the transition matrix
#' @param omega_prior A torch tensor of shape (l, l) containing prior Dirichlet
#'   parameters for the transition matrix
#' @param e_z A torch tensor of shape (i, t, l) containing expected latent
#'   class memberships
#' @param e_zz A torch tensor of shape (i, t-1, l, l) containing expected
#'   transition counts between consecutive time points
#' @param xi A torch tensor of shape (j, l) containing auxiliary variational
#'   parameters for the Jaakkola-Jordan bound
#'
#' @return A scalar value representing the Evidence Lower Bound (ELBO)
#'
#' @details The ELBO is computed as the sum of expected log-likelihood terms
#'   minus KL divergences between posterior and prior distributions. It includes
#'   contributions from item responses, latent class memberships, transitions,
#'   and all model parameters. Higher ELBO values indicate better model fit.
#'
#' @examples
#' \dontrun{
#' # Assuming all required tensors and parameters are available
#' elbo_value <- compute_elbo(
#'   y, delta_mat, m_beta, v_beta,
#'   m_beta_prior, v_beta_prior, alpha, alpha_prior,
#'   omega, omega_prior, e_z, e_zz, xi
#' )
#' }
#'
#' @export
compute_elbo <- function(y, delta_mat, m_beta, v_beta, m_beta_prior,
                         v_beta_prior, alpha, alpha_prior, omega,
                         omega_prior, e_z, e_zz, xi) {
  j <- length(m_beta)

  t <- y$shape[2]

  # Use same device as input tensors
  device <- y$device
  elbo <- torch_zeros(1, dtype = torch_float(), device = device)

  log_phi <- e_log_phi(
    y = y,
    m_beta = m_beta,
    v_beta = v_beta,
    xi = xi,
    delta_mat = delta_mat
  )

  elbo <- elbo + (log_phi * e_z)$sum()

  for (j_iter in seq(j)) {
    tmp <- m_beta[[j_iter]] - m_beta_prior[[j_iter]]

    elbo <- elbo - 1 / 2 * (v_beta[[j_iter]] + tmp$outer(tmp)) |>
      torch_matmul(v_beta_prior[[j_iter]]$inverse()) |>
      torch_trace()

    elbo <- elbo + 1 / 2 * torch_logdet(v_beta[[j_iter]])
  }

  elbo <- elbo + (
    (torch_digamma(alpha) - torch_digamma(alpha$sum())) *
      (alpha_prior - alpha + e_z[, 1, ]$sum(1))
  )$sum()

  elbo <- elbo + (
    (torch_digamma(omega) - torch_digamma(omega$sum(2))$unsqueeze(2)) *
      (omega_prior - omega + e_zz$sum(c(1, 2)))
  )$sum()


  elbo <- elbo + torch_lgamma(alpha)$sum() - torch_lgamma(alpha$sum())

  elbo <- elbo + torch_lgamma(omega)$sum() - torch_lgamma(omega$sum(2))$sum()

  elbo <- elbo - (e_z[, 1, ] * e_z[, 1, ]$log())$sum()

  elbo <- elbo - (
    e_zz * (
      e_zz$log() - e_z[, 1:(t - 1), ]$unsqueeze(4)$log()
    )
  )$sum()

  elbo$item()
}
