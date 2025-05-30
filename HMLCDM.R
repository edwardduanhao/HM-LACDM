source("utils.R")
source("utils_vb.R")
library(cli)
library(zeallot)
library(torch)
library(iterpc)

# HMLCDM_VB <- function(data,
#                            interact = TRUE,
#                            max_iter = 100){
#
# }

max_iter <- 100

# Print the number of threads Torch is using
print(paste("Torch is using", torch_get_num_threads(), "threads"))

# Setup a timer
start <- Sys.time()

# --------------------- Retrieve and format data --------------------- #

# Response matrix Y, shape (N, indT, J)
Y <- torch_tensor(data$Y)

# Number of observations, number of time points, number of items
c(I, indT, J) %<-% Y$shape

# Number of attributes
K <- data$K
L <- 2^K

# Q-matrix
Q_mat <- data$Q_matrix

# Generate the delta matrix for each item
Delta_matrices <- lapply(seq(J), \(j) build_delta(
  q = rep(1, K),
  interact = FALSE
) |>
  torch_tensor())

beta_dim <- rep(K + 1, J)
# --------------------- Parameter initialization --------------------- #

# beta
# M_beta_prior <- lapply(beta_dim, \(d) torch_zeros(d)) # Mean
M_beta <- lapply(beta_dim, \(d) torch_zeros(d)) # Mean
M_beta_prior <- lapply(beta_dim, \(d) torch_tensor(c(-3, rep(0, K)))) # Mean
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
# E_Z <- torch_randn(I, indT, L) |> nnf_softmax(dim = 3)

E_Z_inter <- torch_ones(I, indT, L, L) / (L * L)
E_Z_trace <- torch_zeros(max_iter, I, indT, L) # Trace of Z

# xi
xi <- torch_zeros(J, L)
xi_trace <- torch_zeros(max_iter, J, L) # Trace of xi

# --------------------- Main loop --------------------- #
cli_progress_bar("Running CAVI", total = 100)
for (iter_ in seq(max_iter)){
  
  # Update Z
  log_phi <- E_log_phi(M_beta = M_beta,
                   V_beta = V_beta,
                   xi = xi,
                   Delta_matrices = Delta_matrices)
  
  log_kappa <- E_log_pi(alpha)
  
  log_eta <- E_log_omega(omega)
  
  update_Z(log_phi, log_kappa, log_eta) %->% c(E_Z, E_Z_inter)
  
  # Update beta
  update_beta(
    Y = Y,
    K = K,
    M_beta_prior = M_beta_prior,
    V_beta_prior = V_beta_prior,
    Delta_matrices = Delta_matrices,
    E_Z = E_Z,
    xi = xi
  ) %->% c(M_beta, V_beta)
  
  # Track the trace of beta
  for (j in seq(J)) {
    beta_trace[[j]][iter_, ] <- M_beta[[j]]
  }
  
  
  # Update xi
  update_xi(
    M_beta = M_beta,
    V_beta = V_beta,
    Delta_matrices = Delta_matrices
  ) %->% xi
  
  # Track the trace of xi
  xi_trace[iter_, , ] <- xi
  
  
  # Update omega
  update_omega(
    omega_prior = omega_prior,
    E_Z_inter = E_Z_inter
  ) %->% omega
  
  # Track the trace of omega
  omega_trace[iter_, , ] <- omega
  
  
  # Update alpha
  update_alpha(
    alpha_prior = alpha_prior,
    E_Z = E_Z
  ) %->% alpha
  
  # Track the trace of alpha
  alpha_trace[iter_, ] <- alpha
  cli_progress_update()

}
cli_progress_done()
end_time <- Sys.time()

# --------------------- Post-Hoc Q-matrix Recovery --------------------- #

Q_recovery <- function(M_beta, V_beta, alpha = 0.05, Q_true = NULL){
  # Re-format beta
  beta_hat <- sapply(M_beta, function(x){as_array(x)}) |> t()
  beta_hat_sd <- sapply(V_beta, function(x){sqrt(as_array(torch_diag(x)))}) |> t()

  K <- ncol(beta_hat) - 1

  # Remove the intercept term if all values are negative
  for (i in seq(K+1)){
    if (all(beta_hat[, i] < 0)) {
      cat("Column", i, "is the intercept, will be removed. \n")
      beta_hat_main <- beta_hat[, -i]
      beta_hat_sd_main <- beta_hat_sd[, -i]
      break
    }
  }

  # Perform one-side z-test for each entry
  Z_score <- beta_hat_main / beta_hat_sd_main
  log_p_value <- pnorm(Z_score, log.p = TRUE, lower.tail = FALSE) |> as.vector()
  m <- length(log_p_value)
  o <- order(log_p_value)
  log_p_value_sorted <- log_p_value[o]
  log_q_value <- log(seq(m) / m * alpha) # Benjamini-Hochberg procedure

  n <- max(which(log_p_value_sorted <= log_q_value), 0L)
  sig <- logical(m)
  if (n > 0){
    sig[o[1:n]] <- TRUE
  }
  
  if (!any(sig)) {
    warning("No attribute passed the BH threshold at α = ", alpha, ".")
    Q_hat <- matrix(0L, J, K)
  } else {
    Q_hat <- matrix(sig, nrow = nrow(beta_hat_main), ncol = ncol(beta_hat_main)) * 1L
  }

  if (!is.null(Q_true)) {
    Q_hat_best <- Q_hat
    acc <- 0
    it <- iterpc(K, K, ordered = TRUE)       # permutation iterator  :contentReference[oaicite:1]{index=1}
    repeat {
      if (acc == 1){
        print("Perfect Q-Matrix recovery achieved.")
        break
      }
      idx <- getnext(it, d = 1, drop = TRUE) # vector of column positions
      if (length(idx) == 0){break}            # iterator exhausted
      if (mean((Q_hat[, idx]) == Q_true) > acc){
        acc <- mean((Q_hat[, idx]) == Q_true)
        Q_hat_best <- Q_hat[, idx]
        print(paste0("Permutating ... Best Q-Matrix recovery accuracy is ", round(acc * 100, 3), "%"))
      }
    }
    Q_hat <- Q_hat_best
  }
  return(Q_hat)
}

Q_hat <- Q_recovery(M_beta = M_beta, V_beta = V_beta, Q_true = data$Q_matrix)







