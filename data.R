source("utils.R")

# data generate function for simulation study
data_generate <- function(I, # number of respondents
                          K, # number of latent attributes
                          J, # number of items
                          indT, # number of time points
                          N_dataset = 1, # number of datasets to generate
                          seed = NULL, # seed for reproducibility
                          Q_mat = NULL # generate a random Q-matrix if NULL
) {
  # if the seed is not NULL, set the seed; otherwise, we do not set the seed
  
  if (!is.null(seed)) {
    torch_manual_seed(seed = seed)
  }

  # number of possible attribute profiles
  
  L <- 2^K

  # generate Q-Matrix if it is not given
  
  if (is.null(Q_mat)) {
    Q_mat <- torch_randint(
      low = 1, # inclusive
      high = L, # exclusive
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

  # generate the delta matrix for each item
  
  Delta_matrices <- lapply(seq(J), \(j) build_delta(
    q = Q_mat[j, ],
    interact = FALSE
  ) |>
    torch_tensor())

  # generate the beta vector for each item
  
  beta <- rowSums(Q_mat) |>
    lapply(\(s) build_beta(
      K = s
    ) |>
      torch_tensor())


  # initial distribution of the latent attributes
  
  pii <- torch_ones(L, dtype = torch_float()) / L
  # pii <- nnf_softmax(torch_randn(L, dtype = torch_float()), 1) # random initial distribution

  # transition probability for each attribute
  
  kernel_mat <- matrix(c(0.3, 0.7, 0.2, 0.8),
    nrow = 2,
    byrow = TRUE
  )

  # construct tau matrix from kernel_mat
  
  tau <- torch_ones(L, L, dtype = torch_float())

  for (l_prev in seq(L)) {
    profile_prev <- intToBin(l_prev - 1, d = K)
    for (l_after in seq(L)) {
      profile_after <- intToBin(l_after - 1, d = K)
      for (k in seq(K)) {
        tau[l_prev, l_after] <- tau[l_prev, l_after] *
          kernel_mat[profile_prev[k] + 1, profile_after[k] + 1]
      }
    }
  }

  # Sample the latent attribute profiles over time
  
  int_class <- array(0, c(I, indT))

  # t = 1
  
  int_class[, 1] <- torch_multinomial(pii,
    num_samples = I,
    replacement = TRUE
  ) |>
    as_array()

  # t > 1
  
  for (t in seq(2, indT)) {
    for (l in seq(L)) {
      index <- int_class[, t - 1] == l
      int_class[index, t] <- torch_multinomial(tau[l, ],
        num_samples = as.integer(sum(index)),
        replacement = TRUE
      ) |>
        as_array()
    }
  }

  # generate Response Matrix Y
  
  Y <- torch_zeros(N_dataset, I, indT, J)

  for (t in 1:indT) {
    for (j in 1:J) {
      delta_matrix_t <- Delta_matrices[[j]][as.numeric(int_class[, t]), ]
      Y_sampler <- (delta_matrix_t %@% beta[[j]]) |>
        torch_sigmoid() |>
        distr_bernoulli()
      Y[, , t, j] <- Y_sampler$sample(N_dataset)
    }
  }

  if (N_dataset == 1) {
    Y <- Y[1, , , ]
  }

  # convert int_class to profiles_mat
  
  profiles_mat <- array(0, c(I, indT, K))
  
  for (t in seq(indT)) {
    for (i in seq(I)) {
      profiles_mat[i, t, ] <- intToBin(int_class[i, t] - 1, d = K)
    }
  }

  ground_truth <- list(
    "Q_matrix" = Q_mat,
    "beta" = lapply(beta, \(beta_vec) as_array(beta_vec)),
    "pii" = as_array(pii),
    "tau" = as_array(tau),
    "profiles_index" = int_class,
    "profiles_mat" = profiles_mat
  )

  list(
    "Y" = as_array(Y),
    "K" = K,
    "ground_truth" = ground_truth
  )
}

if (TRUE) {
  Q_mat <- as.matrix(read.table("Q_Matrix/Q_3.txt"))
  # Q_mat <- NULL
  data <- data_generate(
    I = 100,
    K = 3,
    J = 21,
    indT = 3,
    N_dataset = 1,
    seed = 2025,
    Q_mat = Q_mat
  )
}
