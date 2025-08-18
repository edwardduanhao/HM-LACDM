#' Generate Simulated Data for HMLCDM
#'
#' Generates synthetic longitudinal response data for testing and validation of
#' the Hidden Markov Latent Class Diagnostic Model (HMLCDM). This function
#' creates realistic cognitive diagnosis data with temporal dependencies.
#'
#' @param i Number of respondents/examinees
#' @param k Number of latent cognitive attributes
#' @param j Number of assessment items
#' @param t Number of time points in the longitudinal study
#' @param n_dataset Number of datasets to generate (default: 1)
#' @param seed Random seed for reproducibility (default: NULL)
#' @param q_mat Optional Q-matrix specifying item-attribute relationships.
#'   If NULL, a random Q-matrix will be generated (default: NULL)
#' @param device Device for computation. Options: "auto" (default), "cpu", or
#'   "cuda"
#'
#' @return A list containing:
#'   \itemize{
#'     \item y: Response data array, shape (i, t, j)
#'     \item k: Number of attributes (echoed from input)
#'     \item ground_truth: List with true parameters including:
#'       \itemize{
#'         \item q_mat: Item-attribute relationship matrix
#'         \item beta: True item parameters
#'         \item pii: Initial distribution of attribute profiles
#'         \item tau: Transition probabilities between profiles
#'         \item profiles_index: True latent profile indices over time
#'         \item profiles_mat: True binary attribute patterns over time
#'       }
#'   }
#'
#' @examples
#' \dontrun{
#' # Generate data for 100 examinees, 2 attributes, 20 items, 2 time points
#' sim_data <- data_generate(i = 100, k = 2, j = 20, t = 2, seed = 123)
#'
#' # Access response data
#' responses <- sim_data$y
#'
#' # View the true Q matrix
#' print(sim_data$ground_truth$q_mat)
#' }
#'
#' @export
data_generate <- function(
    i, k, j, t, n_dataset = 1, seed = NULL, q_mat = NULL, device = "auto") {
  # Setup device
  device <- get_device(device)

  # Set random seed for reproducibility
  if (!is.null(seed)) {
    torch_manual_seed(seed = seed) # nolint
  }

  # Number of possible attribute profiles
  l <- 2^k

  # Generate Q-Matrix if it is not given
  if (is.null(q_mat)) {
    q_mat <- torch_randint( # nolint
      low = 1, # inclusive
      high = l, # exclusive
      size = j
    ) |>
      as_array() |> # nolint
      sapply(
        \(int_class) {
          int_to_bin(
            x = int_class,
            d = k
          )
        }
      ) |>
      t()
  }

  # Generate the delta matrix for each item
  delta_mat <- lapply(seq(j), \(j_iter) {
    build_delta(
      q = q_mat[j_iter, ],
      interact = FALSE
    ) |>
      torch_tensor(device = device) # nolint
  })

  # Generate the beta vector for each item
  beta <- rowSums(q_mat) |>
    lapply(\(k_iter) {
      build_beta(
        k = k_iter
      ) |>
        torch_tensor(device = device) # nolint
    })


  # Initial distribution of the latent attributes
  pii <- torch_ones(l, dtype = torch_float(), device = device) / l # nolint
  # pii <- nnf_softmax(torch_randn(l, dtype = torch_float(),
  # device = device), 1)

  # Transition probability for each attribute
  kernel_mat <- matrix(c(0.7, 0.3, 0.2, 0.8),
    nrow = 2,
    byrow = TRUE
  )

  # Construct tau matrix from kernel_mat
  tau <- torch_ones(l, l, dtype = torch_float(), device = device) # nolint

  for (l_prev in seq(l)) {
    profile_prev <- int_to_bin(l_prev - 1, d = k)
    for (l_after in seq(l)) {
      profile_after <- int_to_bin(l_after - 1, d = k)
      for (k_iter in seq(k)) {
        tau[l_prev, l_after] <- tau[l_prev, l_after] *
          kernel_mat[profile_prev[k_iter] + 1, profile_after[k_iter] + 1]
      }
    }
  }

  # Sample the latent attribute profiles over time
  int_class <- array(0, c(i, t))

  int_class[, 1] <- torch_multinomial(pii, # nolint
    num_samples = i,
    replacement = TRUE
  ) |>
    as_array() # nolint

  for (t_iter in seq(2, t)) {
    for (l_iter in seq(l)) {
      index <- int_class[, t_iter - 1] == l_iter
      n_samples <- as.integer(sum(index))
      if (n_samples > 0) {
        int_class[index, t_iter] <- torch_multinomial(tau[l_iter, ], # nolint
          num_samples = n_samples,
          replacement = TRUE
        ) |>
          as_array() # nolint
      }
    }
  }

  # Generate Response Matrix Y

  y_mat <- torch_zeros(n_dataset, i, t, j, device = device) # nolint

  for (t_iter in 1:t) {
    for (j_iter in 1:j) {
      delta_mat_t <- delta_mat[[j_iter]][as.numeric(int_class[, t_iter]), ]
      y_sampler <- (delta_mat_t %@% beta[[j_iter]]) |>
        torch_sigmoid() |> # nolint
        distr_bernoulli() # nolint
      y_mat[, , t_iter, j_iter] <- y_sampler$sample(n_dataset)
    }
  }

  if (n_dataset == 1) {
    y_mat <- y_mat[1, , , ]
  }

  # convert int_class to profiles_mat

  profiles_mat <- array(0, c(i, t, k))

  for (t_iter in seq(t)) {
    for (i_iter in seq(i)) {
      profiles_mat[i_iter, t_iter, ] <- int_to_bin(
        x = int_class[i_iter, t_iter] - 1,
        d = k
      )
    }
  }

  ground_truth <- list(
    "q_mat" = q_mat,
    "beta" = lapply(beta, \(beta_vec) as_array(beta_vec)), # nolint
    "pii" = as_array(pii),
    "tau" = as_array(tau),
    "profiles_index" = int_class,
    "profiles_mat" = profiles_mat
  )

  list(
    "y" = as_array(y_mat), # nolint
    "k" = k,
    "ground_truth" = ground_truth
  )
}
