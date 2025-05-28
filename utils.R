require(torch)

# ---------------------------------------------------------------------------------------------------- #

`%@%` <- function(mat1, mat2) {
  if (!inherits(mat1, "torch_tensor") || !inherits(mat2, "torch_tensor")) {
    stop("Both inputs should be torch tensors")
  }
  mat1$matmul(mat2)
}


# ---------------------------------------------------------------------------------------------------- #

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

# Generate the delta matrix for a single item with a given Q-vector
build_delta <- function(q, interact = TRUE) {
  if (sum(q) == 0) {
    stop("At least one attribute should be present")
  }
  
  K <- length(q) # Number of attributes
  
  conversion_matrix <- (seq(2^K) - 1) |> sapply(
    \(int_class) intToBin(
      x = int_class,
      d = K
    )
  ) # Create a conversion matrix of dimension K by 2^K
  
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
    Delta <- Delta[, c(0, 2^seq(from = 0, to = K - 1)) + 1] # Remove interaction terms
  }
  Delta <- Delta[, colSums(Delta) > 0] # Remove columns with all zeros
  Delta
}

# ---------------------------------------------------------------------------------------------------- #

# Generate the beta vector (item parameters), up to 4 attributes (K <= 4)
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

# A function that convert a nested list of tensors to a nested list of arrays
tensor_to_array <- function(nested_list_of_tensors) {
  lapply(nested_list_of_tensors, \(list_of_tensors) {
    lapply(list_of_tensors, as_array)
  })
}

# ---------------------------------------------------------------------------------------------------- #

# A function that convert a nested list of arrays to a nested list of tensors
array_to_tensor <- function(nested_list_of_arrays) {
  lapply(nested_list_of_arrays, \(list_of_arrays) {
    lapply(list_of_arrays, torch_tensor)
  })
}

# ---------------------------------------------------------------------------------------------------- #

# A function that performs batch-wise outer product
batch_outer <- function(mat1, mat2 = NULL) {
  # If mat2 is NULL, perform outer product on mat1
  if(is.null(mat2)) {
    mat2 <- mat1
  }
  # If mat1 is a vector, unsqueeze it to make it a column tensor
  if (length(dim(mat1)) == 1) {
    mat1 <- mat1$unsqueeze(1)
  }
  # If mat2 is a vector, unsqueeze it to make it a row tensor
  if (length(dim(mat2)) == 1) {
    mat2 <- mat2$unsqueeze(1)
  }
  if (length(dim(mat1)) != 2 || length(dim(mat2)) != 2) {
    stop("Both inputs should be 2D tensors")
  }
  ans <- torch_einsum('bi,bj->bij', list(mat1, mat2))
  if (ans$size(1) == 1) {
    return(ans$squeeze(1))
  }
  return(ans)
}

# ---------------------------------------------------------------------------------------------------- #

# A function that vectorizes a tensor by stacking its columns (batch-wise)
vectorize <- function(mat) {
  shape <- mat$shape
  if (length(shape) > 2) {
    return(mat$transpose(-1, -2)$reshape(c(shape[1:(length(shape)-2)], -1)))
  }else if (length(shape) == 2) {
    return(mat$t()$reshape(-1)$unsqueeze(2))
  } else {
    stop("Input should be a tensor with at least 2 dimensions")
  }
}

# ---------------------------------------------------------------------------------------------------- #

# A function that unvectorizes a tensor by reshaping it to the original shape
unvectorize <- function(vec, shape) {
  if (length(shape) > 2) {
    return(vec$reshape(c(shape[1:(length(shape)-2)], shape[length(shape)], shape[length(shape)-1]))$transpose(-1, -2))
  } else if (length(shape) == 2) {
    return(vec$reshape(rev(shape))$t())
  } else {
    stop("Shape should have at least 2 dimensions")
  }
}

# ---------------------------------------------------------------------------------------------------- #

# A function that performs batch-wise Kronecker product
# A (Batch x m x n) and B (Batch x p x q) -> C (Batch x m*p x n*q)
batch_kron <- function(A, B) {
  if (length(dim(A)) == 2 && length(dim(B)) == 2) {
    return(torch_kron(A, B))
  }
  else if(length(dim(A)) == 3 && length(dim(B)) == 2) {
    B <- B$unsqueeze(1)$expand(c(A$size(1), -1, -1))
  }
  else if(length(dim(A)) == 2 && length(dim(B)) == 3) {
    A <- A$unsqueeze(1)$expand(c(B$size(1), -1, -1))
  }
  else if(length(dim(A)) == 3 && length(dim(B)) == 3) {
    if(A$size(1) != B$size(1)) {
      stop("Batch size of A and B should be the same")
    }
  }
  else {
    stop("Both inputs should be 2D or 3D tensors")
  }

  # Get batch size and matrix dimensions
  batch_size <- A$size(1)
  m <- A$size(2)
  n <- A$size(3)
  p <- B$size(2)
  q <- B$size(3)

  # Expand dimensions for broadcasting
  A_exp <- A$unsqueeze(-1)$unsqueeze(-3)  # (batch, m, 1, n, 1)
  B_exp <- B$unsqueeze(-2)$unsqueeze(-4)  # (batch, 1, p, 1, q)

  # Perform elementwise multiplication and reshape
  C <- (A_exp * B_exp)$view(c(batch_size, m * p, n * q))

  return(C)
}






