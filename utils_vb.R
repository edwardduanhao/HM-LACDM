source("utils.R")


# ---------------------------------------------------------------------------------------------------- #

# Jaakkola's function
JJ_func <- function(xi) {
  ans <- (torch_sigmoid(xi) - 1 / 2) / (2 * xi)
  ans[torch_isnan(ans)] <- 1 / 8
  return(ans)
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

  return(list("M_beta" = M_beta, "V_beta" = V_beta))
}

# ---------------------------------------------------------------------------------------------------- #

# updates for the auxiliary variables xi (item-response)
update_xi <- function(M_beta, # list of length J, mean of beta posteriors
                      V_beta, # list of length J, covariance matrix of beta posteriors
                      Delta_matrices # list of length J, Delta matrices
) {
  xi <- torch_zeros(J, 2^K)

  for (j in seq(J)) {
    E_beta2 <- M_beta[[j]]$outer(M_beta[[j]]) + V_beta[[j]]
    xi[j, ] <- (Delta_matrices[[j]] %@% E_beta2 %@% Delta_matrices[[j]]$t())$diagonal()$sqrt()
  }

  return(xi)
}

# ---------------------------------------------------------------------------------------------------- #

# updates for the transition probabilities (omega)
update_omega <- function(omega_prior,
                         E_Z_inter) {
  omega <- omega_prior + E_Z_inter$sum(c(1, 2))
  return(omega)
}

# ---------------------------------------------------------------------------------------------------- #

# updates for the initial distribution of the latent attributes (alpha)
update_alpha <- function(alpha_prior,
                         E_Z) {
  alpha <- alpha_prior + E_Z[, 1, ]$sum(1)
  return(alpha)
}

# ---------------------------------------------------------------------------------------------------- #

# updates for the latent attributes (Z)
update_Z <- function(log_phi,
                     log_kappa,
                     log_eta) {
  c(I, indT, L) %<-% log_phi$shape
  log_f <- torch_zeros(c(I, indT, L), dtype = torch_float())
  log_b <- torch_zeros(c(I, indT, L), dtype = torch_float())
  
  log_f[, 1, ] <- log_kappa + log_phi[, 1, ]
  
  for (t in 2:indT) {
    log_f[ , t, ] <- log_phi[, t, ] + torch_logsumexp(log_f[,t-1,,NULL] + log_eta$transpose(1,2), 2)
    log_b[, indT - t + 1, ] <- torch_logsumexp(log_phi[, indT - t + 2 , , NULL] + log_b[, indT - t + 2, NULL] + log_eta, 2)
  }
  
  E_Z <- nnf_softmax(log_f + log_b, 3)
  temp <- log_f[, 1:(indT-1), , NULL] + log_b[, 2:indT, NULL ,] + log_eta + log_phi[,2:indT,NULL,]
  E_Z_inter <- nnf_softmax(temp$view(c(I, indT-1, -1)), c(3))$view(c(I, indT - 1, L, L))
  
  return(list(E_Z, E_Z_inter))
}

# ---------------------------------------------------------------------------------------------------- #

# Compute the expectation of log(pi)
E_log_pi <- function(alpha) {
  return(torch_digamma(alpha) - torch_digamma(alpha$sum()))
}

# ---------------------------------------------------------------------------------------------------- #

# Compute the expectation of log(omega)
E_log_omega <- function(omega) {
  return(torch_digamma(omega) - torch_digamma(omega$sum(1)))
}

# ---------------------------------------------------------------------------------------------------- #

# Compute the log(phi)
E_log_phi <- function(M_beta,
                      V_beta,
                      xi,
                      Delta_matrices) {
  J <- length(M_beta)
  K <- M_beta[[1]]$shape - 1

  # compute the item-response part
  F_beta <- torch_zeros(J, 2^K)
  F_beta2 <- torch_zeros(J, 2^K)

  for (j in seq(J)) {
    F_beta[j, ] <- Delta_matrices[[j]] %@% M_beta[[j]]
    E_beta2 <- M_beta[[j]]$outer(M_beta[[j]]) + V_beta[[j]]
    F_beta2[j, ] <- (Delta_matrices[[j]] %@% E_beta2 %@% Delta_matrices[[j]]$t())$diagonal()$sqrt()
  }
  log_phi <- ((Y$unsqueeze(-1) - 1 / 2) * F_beta + (nnf_logsigmoid(xi) - xi / 2 - JJ_func(xi) * (F_beta2 - xi$square())))$sum(dim = 3)
  return(log_phi)
}
