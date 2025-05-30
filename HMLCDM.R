source("utils.R")
source("utils_vb.R")
require(cli)
require(zeallot)
require(torch)
require(iterpc)

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
M_beta_prior <- lapply(beta_dim, \(d) torch_zeros(d)) # Mean
# M_beta_prior <- lapply(beta_dim, \(d) torch_tensor(c(-3, rep(3, K)))) # Mean
M_beta <- lapply(beta_dim, \(d) torch_tensor(c(-3, rep(0, K)))) # Mean
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
# xi <- torch_zeros(J, L)
xi <- update_xi(
  M_beta = M_beta,
  V_beta = V_beta,
  Delta_matrices = Delta_matrices
)
xi_trace <- torch_zeros(max_iter, J, L) # Trace of xi
xi_trace[1, , ] <- xi

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


Q_hat <- Q_recovery(M_beta = M_beta, 
                    V_beta = V_beta, 
                    alpha = 0.05,
                    Q_true = data$Q_matrix)







