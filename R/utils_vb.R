source("utils.R")
require(iterpc)

# ---------------------------------------------------------------------------------------------------- #

# Jordan-Jaakkola function
JJ_func <- function(xi, epsilon = 1e-10) {
  x <- torch_abs(xi) + epsilon
  return(torch_tanh(x / 2) / (4 * x))
}

# ---------------------------------------------------------------------------------------------------- #

# updates for the item-response parameters (beta)

update_beta <- function(Y, # tensor, shape = (N, J, indT)
                        K, # integer, number of attributes
                        M_beta_prior, # list of length J, mean of beta priors
                        V_beta_prior, # list of length J, covariance matrix of beta priors
                        Delta_matrices, # list of length J, Delta matrices
                        E_Z, # tensor, shape = (N, 2^(K * indT))
                        xi # tensor, shape = (J, 2^K)
) {
  c(I, indT, J) %<-% Y$shape

  L <- 2^K

  weights_z <- E_Z$sum(c(1, 2))

  M_beta <- vector("list", J)

  V_beta <- vector("list", J)

  for (j in seq(J)) {
    V_beta[[j]] <- torch_inverse(V_beta_prior[[j]]$inverse() +
      2 * Delta_matrices[[j]]$t() %@% torch_diag_embed(weights_z * JJ_func(xi[j, ])) %@% Delta_matrices[[j]])

    M_beta[[j]] <- V_beta[[j]] %@% (V_beta_prior[[j]]$inverse() %@% M_beta_prior[[j]] +
      Delta_matrices[[j]]$t() %@% torch_einsum("ntl,nt->l", list(E_Z, Y[, , j] - 1 / 2)))
  }

  return(list(
    "M_beta" = M_beta,
    "V_beta" = V_beta
  ))
}

# ---------------------------------------------------------------------------------------------------- #

# updates for the auxiliary variables xi (item-response)
update_xi <- function(M_beta, # list of length J, mean of beta posteriors
                      V_beta, # list of length J, covariance matrix of beta posteriors
                      Delta_matrices # list of length J, Delta matrices
) {
  J <- length(M_beta)

  K <- M_beta[[1]]$shape - 1 # number of attributes

  xi <- torch_zeros(J, 2^K)

  for (j in seq(J)) {
    E_beta2 <- M_beta[[j]]$outer(M_beta[[j]]) + V_beta[[j]]

    xi[j, ] <- (Delta_matrices[[j]] %@% E_beta2 %@% Delta_matrices[[j]]$t())$diagonal()$sqrt()
  }

  return(xi)
}

# ---------------------------------------------------------------------------------------------------- #

# updates for the transition probabilities (omega)

update_omega <- function(omega_prior, # tensor, shape = (L, L)
                         E_Z_inter # tensor, shape = (I, indT - 1, L, L)
) {
  return(omega_prior + E_Z_inter$sum(c(1, 2)))
}

# ---------------------------------------------------------------------------------------------------- #

# updates for the initial distribution of the latent attributes (alpha)

update_alpha <- function(alpha_prior, # tensor, shape = (L,)
                         E_Z # tensor, shape = (I, indT, L)
) {
  return(alpha_prior + E_Z[, 1, ]$sum(1))
}

# ---------------------------------------------------------------------------------------------------- #

# updates for the latent attributes (Z)

update_Z <- function(log_phi, # tensor, shape = (I, indT, L)
                     log_kappa, # tensor, shape = (L, )
                     log_eta, # tensor, shape = (L, L),
                     epsilon = 1e-10 # to avoid numerical issues
) {
  c(I, indT, L) %<-% log_phi$shape

  log_f <- torch_zeros(c(I, indT, L), dtype = torch_float())

  log_b <- torch_zeros(c(I, indT, L), dtype = torch_float())

  log_f[, 1, ] <- log_kappa + log_phi[, 1, ]

  for (t in 2:indT) {
    log_f[, t, ] <- log_phi[, t, ] + torch_logsumexp(log_f[, t - 1, , NULL] + log_eta, 2)

    log_b[, indT - t + 1, ] <- torch_logsumexp(log_phi[, indT - t + 2, , NULL] + log_b[, indT - t + 2, , NULL] + log_eta, 2)
  }

  E_Z <- nnf_softmax(log_f + log_b, 3) + epsilon # add epsilon to avoid numerical issues

  E_Z <- E_Z / E_Z$sum(dim = 3, keepdim = TRUE) # renormalise

  temp <- log_f[, 1:(indT - 1), , NULL] + log_b[, 2:indT, NULL, ] + log_eta + log_phi[, 2:indT, NULL, ]

  E_Z_inter <- nnf_softmax(temp$view(c(I, indT - 1, -1)), 3) + epsilon # add epsilon to avoid numerical issues

  E_Z_inter <- E_Z_inter / E_Z_inter$sum(dim = 3, keepdim = TRUE) # renormalise

  E_Z_inter <- E_Z_inter$view(c(I, indT - 1, L, L))

  return(list(E_Z, E_Z_inter))
}

# ---------------------------------------------------------------------------------------------------- #

# compute the expectation of log(pi)

E_log_pi <- function(alpha # tensor, shape = (L, )
) {
  return(torch_digamma(alpha) - torch_digamma(alpha$sum()))
}

# ---------------------------------------------------------------------------------------------------- #

# compute the expectation of log(omega)

E_log_omega <- function(omega # tensor, shape = (L, L)
) {
  return(torch_digamma(omega) - torch_digamma(omega$sum(1)))
}

# ---------------------------------------------------------------------------------------------------- #

# compute the log(phi)

E_log_phi <- function(Y, # tensor, shape = (I, indT, J)
                      M_beta, # list of length J, mean of beta posteriors
                      V_beta, # list of length J, covariance matrix of beta posteriors
                      xi, # tensor, shape = (J, L)
                      Delta_matrices # list of length J, Delta matrices
) {
  J <- length(M_beta)

  K <- M_beta[[1]]$shape - 1

  # compute the item-response part

  F_beta <- torch_zeros(J, 2^K)

  F_beta2 <- torch_zeros(J, 2^K)

  for (j in seq(J)) {
    F_beta[j, ] <- Delta_matrices[[j]] %@% M_beta[[j]]

    E_beta2 <- M_beta[[j]]$outer(M_beta[[j]]) + V_beta[[j]]

    F_beta2[j, ] <- (Delta_matrices[[j]] %@% E_beta2 %@% Delta_matrices[[j]]$t())$diagonal()
  }

  log_phi <- ((Y$unsqueeze(-1) - 1 / 2) * F_beta + (nnf_logsigmoid(xi) - xi / 2 - JJ_func(xi) * (F_beta2 - xi$square())))$sum(dim = 3)

  return(log_phi)
}

# ------------------------------------------------------------------------------------------------------ #

# post-hoc Q-matrix recovery

Q_recovery <- function(M_beta, # list of length J, mean of beta posteriors
                       V_beta, # list of length J, covariance matrix of beta posteriors
                       beta_hat_trace, # matrix, shape = (max_iter, J, K + 1)
                       E_Z, # tensor, shape = (I, indT, L)
                       alpha_level = 0.05, # significance level
                       Q_true = NULL # array, shape = (J, K)
) {
  # re-format beta

  beta_hat <- sapply(M_beta, \(x) as_array(x)) |> t()

  beta_hat_sd <- sapply(V_beta, \(x) sqrt(as_array(torch_diag(x)))) |> t()

  beta_hat_trace <- torch_stack(beta_hat_trace) |>
    torch_transpose(1, 2) |>
    as_array()

  K <- ncol(beta_hat) - 1

  # remove the intercept term if all values are negative

  intercept_detected <- FALSE

  for (i in seq(K + 1)) {
    if (all(beta_hat[, i] < 0)) {
      cat("Column", i, "is the intercept, will be removed. \n")

      beta_hat_main <- beta_hat[, -i]

      beta_hat_sd_main <- beta_hat_sd[, -i]

      beta_hat_trace_main <- beta_hat_trace[, , -i]

      intercept_detected <- TRUE

      beta_hat_intercept <- beta_hat[, i]

      beta_hat_sd_intercept <- beta_hat_sd[, i]

      beta_hat_trace_intercept <- beta_hat_trace[, , i]

      break
    }
  }

  if (!intercept_detected) {
    stop("No intercept term detected.")
  }

  # perform one-side z-test for each entry
  Z_score <- beta_hat_main / beta_hat_sd_main

  log_p_value <- pnorm(Z_score, log.p = TRUE, lower.tail = FALSE) |> as.vector()

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
    warning("No attribute passed the BH threshold at α = ", alpha_level, ".")

    Q_hat <- matrix(0L, J, K)
  } else {
    Q_hat <- matrix(sig, nrow = nrow(beta_hat_main), ncol = ncol(beta_hat_main)) * 1L
  }

  if (!is.null(Q_true)) {
    Q_hat_best <- Q_hat

    idx_best <- NULL

    acc <- 0

    it <- iterpc(K, K, ordered = TRUE) # permutation iterator  :contentReference[oaicite:1]{index=1}

    repeat {
      if (acc == 1) {
        print("Perfect Q-Matrix recovery achieved.")
        break
      }

      idx <- getnext(it, d = 1, drop = TRUE) # vector of column positions

      if (length(idx) == 0) {
        break
      } # iterator exhausted
      if (mean((Q_hat[, idx]) == Q_true) > acc) {
        acc <- mean((Q_hat[, idx]) == Q_true)

        Q_hat_best <- Q_hat[, idx]

        idx_best <- idx

        print(paste0("Permutating ... Best Q-Matrix recovery accuracy is ", round(acc * 100, 3), "%"))
      }
    }


    beta_hat_permuted <- cbind(beta_hat_intercept, beta_hat_main[, idx_best])
    
    colnames(beta_hat_permuted) <- NULL

    beta_hat_sd_permuted <- cbind(beta_hat_sd_intercept, beta_hat_sd_main[, idx_best])
    
    colnames(beta_hat_sd_permuted) <- NULL

    beta_hat_trace_permuted <- array(dim = dim(beta_hat_trace))

    beta_hat_trace_permuted[, , 1] <- beta_hat_trace_intercept

    beta_hat_trace_permuted[, , 2:(K + 1)] <- beta_hat_trace_main[, , idx_best]

    Q_hat <- Q_hat_best

    I <- E_Z$shape[1]

    indT <- E_Z$shape[2]

    profiles_index <- sapply(seq(indT), \(t)
    sapply(seq(I), \(i) {
      sum(intToBin(as.numeric(E_Z[i, t, ]$argmax()) - 1, K) * 2^(idx_best - 1)) + 1
    }))

    return(list(
      "Q_hat" = Q_hat,
      "acc" = acc,
      "beta_hat" = beta_hat_permuted,
      "beta_hat_sd" = beta_hat_sd_permuted,
      "beta_hat_trace" = beta_hat_trace_permuted,
      "profiles_index" = profiles_index,
      "ord" = idx_best
    ))
  }
  return(Q_hat)
}

# ------------------------------------------------------------------------------------------------------ #

# compute the evidence lower bound (ELBO)

compute_elbo <- function(Y, # tensor, shape = (I, indT, J)
                         Delta_matrices, # list of length J, Delta matrices
                         M_beta, # list of length J, mean of beta posteriors
                         V_beta, # list of length J, covariance matrix of beta posteriors
                         M_beta_prior, # list of length J, mean of beta priors
                         V_beta_prior, # list of length J, covariance matrix of beta priors
                         alpha, # tensor, shape = (L,)
                         alpha_prior, # tensor, shape = (L,)
                         omega, # tensor, shape = (L,L)
                         omega_prior, # tensor, shape = (L,L)
                         E_Z, # tensor, shape = (I, indT, L)
                         E_Z_inter, # tensor, shape = (I, indT-1, L,L)
                         xi # tensor, shape = (J, L)
) {
  J <- length(M_beta)

  indT <- Y$shape[2]

  elbo <- torch_zeros(1, dtype = torch_float())

  log_phi <- E_log_phi(Y,
    M_beta = M_beta,
    V_beta = V_beta,
    xi = xi,
    Delta_matrices = Delta_matrices
  )

  elbo <- elbo + (log_phi * E_Z)$sum()

  for (j in seq(J)) {
    elbo <- elbo - 1 / 2 * (V_beta[[j]] + (M_beta[[j]] - M_beta_prior[[j]])$outer((M_beta[[j]] - M_beta_prior[[j]]))) |>
      torch_matmul(V_beta_prior[[j]]$inverse()) |>
      torch_trace()

    elbo <- elbo + 1 / 2 * torch_logdet(V_beta[[j]])
  }

  elbo <- elbo + ((torch_digamma(alpha) - torch_digamma(alpha$sum())) * (alpha_prior - alpha + E_Z[, 1, ]$sum(1)))$sum()

  elbo <- elbo + ((torch_digamma(omega) - torch_digamma(omega$sum(2))$unsqueeze(2)) * (omega_prior - omega + E_Z_inter$sum(c(1, 2))))$sum()

  elbo <- elbo + torch_lgamma(alpha)$sum() - torch_lgamma(alpha$sum())

  elbo <- elbo + torch_lgamma(omega)$sum() - torch_lgamma(omega$sum(2))$sum()

  elbo <- elbo - (E_Z[, 1, ] * E_Z[, 1, ]$log())$sum()

  elbo <- elbo - (E_Z_inter * (E_Z_inter$log() - E_Z[, 1:(indT - 1), ]$unsqueeze(4)$log()))$sum()

  return(elbo$item())
}
