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
#' int_to_bin(5, 4)  # Returns c(1, 0, 1, 0)
#' int_to_bin(0, 3)  # Returns c(0, 0, 0)
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
#' q_vector <- c(1, 1, 0)  # Item requires attributes 1 and 2
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
#' build_beta(2)  # Returns c(-3, 3, 3)
#' build_beta(4)  # Returns c(-3, 1.5, 1.5, 1.5, 1.5)
#'
#' @export
build_beta <- function(k) {
  switch(as.character(k),
    "1" = c(-3, 6),
    "2" = c(-3, 3, 3),
    "3" = c(-3, 2, 2, 2),
    "4" = c(-3, 1.5, 1.5, 1.5, 1.5),
    "5" = c(-3, 1, 1, 1, 1, 1),
    stop("Invalid K")
  )
}

# ---------------------------------------------------------------------------- #
