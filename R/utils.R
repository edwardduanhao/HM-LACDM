# ---------------------------------------------------------------------------- #

#' Configure Device for CUDA Support
#'
#' Automatically selects the best available device (CUDA or CPU) for
#' computation, or uses a user-specified device with fallback handling.
#'
#' @param device Character string specifying the device preference. Options are:
#'   \itemize{
#'     \item "auto" (default): Automatically select CUDA if available,
#'           otherwise CPU
#'     \item "cuda": Force CUDA usage (falls back to CPU if unavailable)
#'     \item "cpu": Force CPU usage
#'   }
#'
#' @return Character string indicating the selected device ("cuda" or "cpu")
#'
#' @examples
#' \dontrun{
#' device <- get_device("auto")
#' device <- get_device("cuda")
#' }
#'
#' @export
get_device <- function(device = "auto") {
  if (device == "auto") {
    if (cuda_is_available()) { # nolint
      device <- "cuda"
      cat("CUDA is available. Using GPU acceleration.\n")
    } else {
      device <- "cpu"
      cat("CUDA not available. Using CPU.\n")
    }
  } else if (device == "cuda" && !cuda_is_available()) {
    warning("CUDA requested but not available. Falling back to CPU.")
    device <- "cpu"
  }

  return(device)
}

# ---------------------------------------------------------------------------- #

#' Matrix Multiplication Operator for Torch Tensors
#'
#' A custom infix operator that performs matrix multiplication between two torch
#' tensors. This is a convenience function that wraps the 'torch_matmul'
#' function.
#'
#' @param mat1 A torch tensor (left operand)
#' @param mat2 A torch tensor (right operand)
#'
#' @return A torch tensor resulting from the matrix multiplication of mat1 and
#'   mat2
#'
#' @examples
#' \dontrun{
#' a <- torch_randn(3, 4)
#' b <- torch_randn(4, 5)
#' result <- a %@% b
#' }
#'
#' @export
`%@%` <- function(mat1, mat2) {
  if (!inherits(mat1, "torch_tensor") || !inherits(mat2, "torch_tensor")) {
    stop("Both inputs should be torch tensors")
  }
  mat1$matmul(mat2)
}

# ---------------------------------------------------------------------------- #

#' Convert Integer to Binary Vector
#'
#' Converts an integer to its binary representation as a vector of 0s and 1s
#' with a specified length.
#'
#' @param x An integer to convert (must be in range [0, 2^d - 1])
#' @param d The desired length of the binary vector
#'
#' @return A numeric vector of length d containing the binary representation
#'   of x
#'
#' @examples
#' int_to_bin(5, 4) # Returns c(1, 0, 1, 0)
#' int_to_bin(0, 3) # Returns c(0, 0, 0)
#'
#' @export
int_to_bin <- function(x, d) {
  if (x < 0 || x >= 2^d) {
    stop("x should be in [0, 2^d - 1]")
  }

  x |>
    intToBits() |>
    as.integer() |>
    (\(bits) bits[1:d])()
}

# ---------------------------------------------------------------------------- #

#' Build Delta Matrix for CDM Item
#'
#' Generates the delta matrix for a single item in a Cognitive Diagnosis Model
#' (CDM) based on the item's Q-vector (attribute requirements).
#'
#' @param q A binary vector indicating which attributes are required for the
#'   item (Q-vector)
#' @param interact Logical indicating whether to include interaction terms
#'   (default: FALSE)
#'
#' @return A matrix representing the delta matrix for the item, where rows
#'   correspond to attribute profiles and columns to item parameters
#'
#' @examples
#' \dontrun{
#' q_vector <- c(1, 1, 0) # Item requires attributes 1 and 2
#' delta_matrix <- build_delta(q_vector)
#' }
#'
#' @export
build_delta <- function(q, interact = FALSE) {
  if (sum(q) == 0) {
    stop("At least one attribute should be present")
  }

  k <- length(q) # Number of attributes

  conversion_matrix <- (seq(2^k) - 1) |> sapply(
    \(int_class) {
      int_to_bin(
        x = int_class,
        d = k
      )
    }
  ) # Create a conversion matrix of dimension k by 2^k

  q_mat <- conversion_matrix |>
    apply(MARGIN = 2, FUN = \(vec) prod(q^vec)) |>
    rep(times = 2^k) |>
    matrix(
      nrow = 2^k,
      byrow = TRUE
    )

  a_mat <- conversion_matrix |> # nolint
    apply(2, \(vec1) {
      conversion_matrix |>
        apply(2, \(vec2) prod(vec2^vec1))
    })

  delta <- a_mat * q_mat

  if (!interact) {
    # Remove interaction terms
    delta <- delta[, c(0, 2^seq(from = 0, to = k - 1)) + 1]
  }
  delta <- delta[, colSums(delta) > 0] # Remove columns with all zeros
  delta
}

# ---------------------------------------------------------------------------- #

#' Build the beta vector (item parameters), up to 5 attributes (k <= 5)
#'
#' Generates predefined beta (item parameter) vectors for items in a Cognitive
#' Diagnosis Model based on the number of required attributes.
#'
#' @param k Number of attributes required by the item (at most 5)
#'
#' @return A numeric vector of item parameters (beta values)
#'
#' @examples
#' build_beta(2) # Returns c(-3, 3, 3)
#' build_beta(4) # Returns c(-3, 1.5, 1.5, 1.5, 1.5)
#'
#' @export
build_beta <- function(k, mode = "strong") {
  if (mode == "strong") {
    beta <- c(-3, rep(6 / k, k))
  } else if (mode == "moderate") {
    beta <- c(-2, rep(4 / k, k))
  } else if (mode == "weak") {
    beta <- c(-1.5, rep(3 / k, k))
  } else {
    stop("mode should be one of 'strong', 'moderate', or 'weak'")
  }

  beta
}

# ---------------------------------------------------------------------------- #

#' Plot the trajectory of alpha parameters over iterations
#'
#' @param alpha_trace A matrix containing the alpha parameter traces
#' @param iter_trunc The number of iterations to truncate the plot (default: 30)
#' @param save Logical indicating whether to save the plot (default: TRUE)
#' @param path The directory path to save the plot (default: "inst/figures/singlerun/")
#' @param file_name The name of the file to save the plot (default: "alpha_trace.pdf")
#'
#' @return The ggplot object for the alpha trace plot
#'
#' @examples
#' alpha_trace <- matrix(rnorm(300), ncol = 3)
#' plot_alpha_trace(alpha_trace)
#'
#' @export
plot_alpha_trace <- function(
    alpha_trace,
    iter_trunc = 30,
    save = TRUE,
    path = "inst/figures/singlerun/",
    file_name = "alpha_trace.pdf") {
  df_alpha <- as.data.frame(alpha_trace[seq(iter_trunc), ]) |> stack()

  df_alpha$iter <- rep(seq(iter_trunc), times = ncol(alpha_trace))

  names(df_alpha) <- c("value", "variable", "iter")

  fig_alpha <- ggplot(df_alpha, aes(x = iter, y = value, group = variable)) +
    geom_line(color = "#2774AE") +
    theme_classic() +
    theme(legend.position = "none") +
    labs(x = "Iteration", y = expression(alpha))

  if (save) {
    ggsave(
      filename = file.path(path, file_name),
      plot = fig_alpha,
      width = 6,
      height = 4,
      units = "in",
      dpi = 600
    )
  }
  fig_alpha
}

# ---------------------------------------------------------------------------- #

#' Plot the trajectory of beta parameters over iterations
#'
#' @param beta_trace A matrix containing the beta parameter traces
#' @param iter_trunc The number of iterations to truncate the plot (default: 30)
#' @param save Logical indicating whether to save the plot (default: TRUE)
#' @param path The directory path to save the plot (default: "inst/figures/singlerun/")
#' @param file_name The name of the file to save the plot (default: "beta_trace.pdf")
#'
#' @return The ggplot object for the beta trace plot
#'
#' @examples
#' beta_trace <- matrix(rnorm(300), ncol = 3)
#' plot_beta_trace(beta_trace)
#'
#' @export
plot_beta_trace <- function(
    beta_trace,
    beta_true,
    iter_trunc = 30,
    save = TRUE,
    path = "inst/figures/singlerun/",
    file_name = "beta_trace.pdf") {
  df_beta <- matrix(beta_trace,
    nrow = dim(beta_trace)[1],
    ncol = prod(dim(beta_trace)[-1])
  )[seq(iter_trunc), ] |>
    as.data.frame() |>
    stack()

  df_beta$iter <- rep(seq(iter_trunc), times = ncol(beta_trace))

  df_beta$type <- rep(as.factor(beta_true), each = iter_trunc)

  df_beta$iter <- rep(seq(iter_trunc), times = ncol(beta_trace))

  names(df_beta) <- c("value", "variable", "iter", "type")

  fig_beta <- ggplot(
    df_beta,
    aes(x = iter, y = value, group = variable, color = type)
  ) +
    geom_line() +
    theme_classic() +
    theme(legend.position = "none") +
    labs(x = "Iteration", y = expression(beta)) +
    scale_color_brewer(palette = "Set1")

  if (save) {
    ggsave(
      filename = file.path(path, file_name),
      plot = fig_beta,
      width = 6,
      height = 4,
      units = "in",
      dpi = 600
    )
    fig_beta
  }
}


# ---------------------------------------------------------------------------- #

#' Plot the trajectory of omega parameters over iterations
#'
#' @param omega_trace A matrix containing the omega parameter traces
#' @param iter_trunc The number of iterations to truncate the plot (default: 30)
#' @param save Logical indicating whether to save the plot (default: TRUE)
#' @param path The directory path to save the plot (default: "inst/figures/singlerun/")
#' @param file_name The name of the file to save the plot (default: "omega_trace.pdf")
#'
#' @return The ggplot object for the omega trace plot
#'
#' @examples
#' omega_trace <- matrix(rnorm(300), ncol = 3)
#' plot_omega_trace(omega_trace)
#'
#' @export
plot_omega_trace <- function(
    omega_trace,
    iter_trunc = 30,
    save = TRUE,
    path = "inst/figures/singlerun/",
    file_name = "omega_trace.pdf") {
  df_omega <- matrix(omega_trace,
    nrow = dim(omega_trace)[1],
    ncol = prod(dim(omega_trace)[-1])
  )[seq(iter_trunc), ] |>
    as.data.frame() |>
    stack()

  df_omega$iter <- rep(seq(iter_trunc), times = ncol(omega_trace))

  names(df_omega) <- c("value", "variable", "iter")

  fig_omega <- ggplot(df_omega, aes(x = iter, y = value, group = variable)) +
    geom_line(color = "#2774AE") +
    theme_classic() +
    theme(legend.position = "none") +
    labs(x = "Iteration", y = expression(omega))

  if (save) {
    ggsave(
      filename = file.path(path, file_name),
      plot = fig_omega,
      width = 6,
      height = 4,
      units = "in",
      dpi = 600
    )
  }
  fig_omega
}


# ---------------------------------------------------------------------------- #

#' Plot the trajectory of elbo over iterations
#'
#' @param elbo_trace A vector containing the elbo values
#' @param iter_trunc The number of iterations to truncate the plot (default: 30)
#' @param save Logical indicating whether to save the plot (default: TRUE)
#' @param path The directory path to save the plot (default: "inst/figures/singlerun/")
#' @param file_name The name of the file to save the plot (default: "elbo_trace.pdf")
#'
#' @return The ggplot object for the elbo trace plot
#'
#' @examples
#' elbo_trace <- rnorm(100)
#' plot_elbo_trace(elbo_trace)
#'
#' @export
plot_elbo_trace <- function(
    elbo_trace,
    iter_trunc = 30,
    save = TRUE,
    path = "inst/figures/singlerun/",
    file_name = "elbo_trace.pdf") {
  df_elbo <- data.frame(
    iteration = seq(iter_trunc),
    elbo = elbo_trace[seq(iter_trunc)]
  )

  fig_elbo <- ggplot(df_elbo, aes(x = iteration, y = elbo)) +
    geom_line(linewidth = 1) +
    geom_point(size = 2) +
    labs(x = "Iteration", y = "Evidence Lower Bound") +
    theme_classic()

  if (save) {
    ggsave(
      filename = file.path(path, file_name),
      plot = fig_elbo,
      width = 6,
      height = 4,
      units = "in",
      dpi = 600
    )
  }
  fig_elbo
}


# ---------------------------------------------------------------------------- #

#' Plot the recovery plot of tau
#'
#' @param tau_hat A vector containing the estimated tau values
#' @param tau_true A vector containing the true tau values
#' @param save Logical indicating whether to save the plot (default: TRUE)
#' @param path The directory path to save the plot (default: "inst/figures/singlerun/")
#' @param file_name The name of the file to save the plot (default: "tau_recovery.pdf")
#'
#' @return The ggplot object for the tau recovery plot
#'
#' @examples
#' tau_hat <- rnorm(100)
#' tau_true <- rnorm(100)
#' plot_tau_recovery(tau_hat, tau_true)
#'
#' @export
plot_tau_recovery <- function(
    tau_hat,
    tau_true,
    save = TRUE,
    path = "inst/figures/singlerun/",
    file_name = "tau_recovery.pdf") {
  df_tau_recovery <- data.frame(
    tau_true = as.vector(tau_true),
    tau_hat = as.vector(tau_hat)
  )

  fig_tau_recovery <- ggplot(df_tau_recovery, aes(x = tau_true, y = tau_hat)) +
    geom_point() +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    theme_minimal() +
    labs(
      x = expression(paste("True ", tau)),
      y = expression(paste("Estimated ", tau))
    )
  if (save) {
    ggsave(
      filename = file.path(path, file_name),
      plot = fig_tau_recovery,
      width = 6,
      height = 4,
      units = "in",
      dpi = 600
    )
  }
  fig_tau_recovery
}


# ---------------------------------------------------------------------------- #

#' Plot the recovery plot of pi
#'
#' @param pii_hat A vector containing the estimated pi values
#' @param pii_true A vector containing the true pi values
#' @param save Logical indicating whether to save the plot (default: TRUE)
#' @param path The directory path to save the plot (default: "inst/figures/singlerun/")
#' @param file_name The name of the file to save the plot (default: "pii_recovery.pdf")
#'
#' @return The ggplot object for the pi recovery plot
#'
#' @examples
#' pii_hat <- rnorm(100)
#' pii_true <- rnorm(100)
#' plot_pii_recovery(pii_hat, pii_true)
#'
#' @export
plot_pii_recovery <- function(
    pii_hat,
    pii_true,
    save = TRUE,
    path = "inst/figures/singlerun/",
    file_name = "pii_recovery.pdf") {
  df_pii_recovery <- data.frame(
    pii_true = as.vector(pii_true),
    pii_hat = as.vector(pii_hat)
  )

  fig_pii_recovery <- ggplot(df_pii_recovery, aes(x = pii_true, y = pii_hat)) +
    geom_point() +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    theme_minimal() +
    labs(
      x = expression(paste("True ", pi)),
      y = expression(paste("Estimated ", pi))
    )

  if (save) {
    ggsave(
      filename = file.path(path, file_name),
      plot = fig_pii_recovery,
      width = 6,
      height = 4,
      units = "in",
      dpi = 600
    )
  }
  fig_pii_recovery
}


# ---------------------------------------------------------------------------- #

#' Plot the recovery plot of beta
#'
#' @param pii_hat A vector containing the estimated pi values
#' @param pii_true A vector containing the true pi values
#' @param save Logical indicating whether to save the plot (default: TRUE)
#' @param path The directory path to save the plot (default: "inst/figures/singlerun/")
#' @param file_name The name of the file to save the plot (default: "pii_recovery.pdf")
#'
#' @return The ggplot object for the pi recovery plot
#'
#' @examples
#' pii_hat <- rnorm(100)
#' pii_true <- rnorm(100)
#' plot_pii_recovery(pii_hat, pii_true)
#'
#' @export
plot_beta_recovery <- function(
    beta_hat,
    beta_true,
    save = TRUE,
    path = "inst/figures/singlerun/",
    file_name = "beta_recovery.pdf") {
  df_beta_recovery <- data.frame(
    beta_true = as.vector(beta_true),
    beta_hat = as.vector(beta_hat)
  )

  fig_beta_recovery <- ggplot(
    df_beta_recovery,
    aes(x = beta_true, y = beta_hat)
  ) +
    geom_point() +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    theme_minimal() +
    labs(
      x = expression(paste("True ", beta)),
      y = expression(paste("Estimated ", beta))
    )

  if (save) {
    ggsave(
      filename = file.path(path, file_name),
      plot = fig_beta_recovery,
      width = 6,
      height = 4,
      units = "in",
      dpi = 600
    )
  }
  fig_beta_recovery
}
