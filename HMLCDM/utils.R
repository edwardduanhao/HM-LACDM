require(torch)
require(zeallot)

# ---------------------------------------------------------------------------------------------------- #

# matrix multiplication operator for torch tensors

`%@%` <- function(mat1, mat2) {
  if (!inherits(mat1, "torch_tensor") || !inherits(mat2, "torch_tensor")) {
    stop("Both inputs should be torch tensors")
  }
  mat1$matmul(mat2)
}

# ---------------------------------------------------------------------------------------------------- #

# convert an integer to a binary vector of length d

intToBin <- function(x, d) {
  if (x < 0 || x >= 2^d) {
    stop("x should be in [0, 2^d - 1]")
  }

  x |>
    intToBits() |>
    as.integer() |>
    (\(bits) bits[1:d])()
}

# ---------------------------------------------------------------------------------------------------- #

# generate the delta matrix for a single item with a given Q-vector

build_delta <- function(q, interact = FALSE) {
  if (sum(q) == 0) {
    stop("At least one attribute should be present")
  }

  K <- length(q) # number of attributes

  conversion_matrix <- (seq(2^K) - 1) |> sapply(
    \(int_class) intToBin(
      x = int_class,
      d = K
    )
  ) # create a conversion matrix of dimension K by 2^K

  Q <- conversion_matrix |>
    apply(MARGIN = 2, FUN = \(vec) prod(q^vec)) |>
    rep(times = 2^K) |>
    matrix(
      nrow = 2^K,
      byrow = TRUE
    )

  A <- conversion_matrix |>
    apply(2, \(vec1)
    conversion_matrix |>
      apply(2, \(vec2) prod(vec2^vec1)))

  Delta <- A * Q

  if (!interact) {
    Delta <- Delta[, c(0, 2^seq(from = 0, to = K - 1)) + 1] # remove interaction terms
  }
  Delta <- Delta[, colSums(Delta) > 0] # remove columns with all zeros
  Delta
}

# ---------------------------------------------------------------------------------------------------- #

# generate the beta vector (item parameters), up to 4 attributes (K <= 4)

build_beta <- function(K) {
  switch(as.character(K),
    "1" = c(-3, 6),
    "2" = c(-3, 3, 3),
    "3" = c(-3, 2, 2, 2),
    "4" = c(-3, 1.5, 1.5, 1.5, 1.5),
    stop("Invalid K")
  )
}

# ---------------------------------------------------------------------------------------------------- #

