
# Generate simulated data
set.seed(123)
data <- data_generate(
  I = 2000,       # number of respondents
  K = 2,         # number of latent attributes
  J = 21,        # number of items
  indT = 2,      # number of time points
  seed = 123,
  device = "auto"  # automatically use CUDA if available
)

# Run HMLCDM with automatic device selection
result <- HMLCDM_VB(
  data = data,
  max_iter = 100,
  alpha_level = 0.05,
  elbo = TRUE,
  device = "cuda"  # "auto", "cpu", or "cuda"
)

result1 <- HMLCDM_VB(
  data = data,
  max_iter = 100,
  alpha_level = 0.05,
  elbo = TRUE,
  device = "cpu"
)