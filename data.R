library(torch)
library(zeallot)
source("utils.R")

# Data generate function for simulation study
data_generate <- function(I, # Number of respondents
                          K, # Number of latent attributes
                          J, # Number of items
                          indT, # Number of time points
                          N_dataset = 1, # Number of datasets to generate
                          seed = NULL, # Seed for reproducibility
                          Q_mat = NULL # Generate a random Q-matrix if NULL
) {
  
  # If the seed is not NULL, set the seed; otherwise, we do not set the seed
  if (!is.null(seed)) {
    torch_manual_seed(seed = seed)
  }
  
  # Generate Q-Matrix if it is not given
  if (is.null(Q_mat)) {
    Q_mat <- torch_randint(
      low = 1, # inclusive
      high = 2^K, # exclusive
      size = J
    ) |>
      as_array() |>
      sapply(
        \(int_class) intToBin(
          x = int_class,
          d = K
        )
      ) |>
      t()
  }
  
  # Generate the delta matrix for each item
  Delta_matrices <- lapply(seq(J), \(j) build_delta(
    q = Q_mat[j, ],
    interact = FALSE
  ) |>
    torch_tensor())
  
  # Generate the beta vector for each item
  beta <- rowSums(Q_mat) |>
    lapply(\(s) build_beta(
      K = s
    ) |>
      torch_tensor())
  
  
  # Initial distribution of the latent attributes
  pii <- torch_ones(2^K, dtype = torch_float()) / 2^K
  
  # Transition probability
  kernel_mat <- matrix(c(0.4, 0.6, 0.1, 0.9), nrow = 2, byrow = TRUE)
  
  omega <- torch_ones(2^K, 2^K, dtype = torch_float())
  
  for (l_prev in seq(2^K)){
    profile_prev <- intToBin(l_prev - 1, d = K)
    for (l_after in seq(2^K)){
      profile_after <- intToBin(l_after - 1, d = K)
      for (k in seq(K)){
        omega[l_prev, l_after] <- omega[l_prev, l_after] * 
          kernel_mat[profile_prev[k] + 1, profile_after[k] + 1]
      }
    }
  }
  
  # Sample the latent attribute profiles over time
  int_class <- array(0, c(I, indT))
  
  # t = 1
  int_class[,1] <- as_array(torch_multinomial(pii, I, replacement=TRUE))
  
  # t > 1
  for (t in seq(2, indT)){
    for (l in seq(2^K)){
      index <- int_class[, t - 1] == l
      int_class[index, t] <- as_array(torch_multinomial(omega[l, ], as.integer(sum(index)), replacement = TRUE))
    }
  }
  
  # Generate Response Matrix Y
  Y <- torch_zeros(N_dataset, I, indT, J)
  
  for (t in 1:indT) {
    for (j in 1:J) {
      delta_matrix_t <- Delta_matrices[[j]][as.numeric(int_class[,t]), ]
      Y_sampler <- (delta_matrix_t %@% beta[[j]]) |> torch_sigmoid() |> distr_bernoulli()
      Y[, , t, j] <- Y_sampler$sample(N_dataset)
    }
  }

  if (N_dataset == 1) {
    Y <- Y[1, , , ]
  }
  
  # Convert int_class to profiles_mat
  profiles_mat <- array(0, c(I, indT, K))
  for (t in seq(indT)) {
    for (i in seq(I)) {
      profiles_mat[i, t, ] <- intToBin(int_class[i, t] - 1, d = K)
    }
  }
  
  list(
    "Y" = as_array(Y),
    "profiles_mat" = profiles_mat,
    "profiles_index" = int_class,
    "beta" = lapply(beta, \(beta_vec) as_array(beta_vec)),
    "Q_matrix" = Q_mat,
    "folder_name" = paste("I", I, "K", K, "J", J, "T", indT, sep = "_")
  )
}

if (TRUE) {
  Q_mat <- as.matrix(read.table("Q_Matrix/Q_3.txt"))
  data <- data_generate(
    I = 1000,
    K = 3,
    J = 21,
    indT = 2,
    N_dataset = 1,
    seed = 2025,
    Q_mat = Q_mat
  )
}

print("Hi")






