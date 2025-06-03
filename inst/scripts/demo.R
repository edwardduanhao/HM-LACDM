library(ggplot2)
library(dplyr)

# load the Q-matrix

Q_mat <- as.matrix(read.table("inst/Q_Matrix/Q_3.txt"))

# generate data

data <- data_generate(
  I = 200, # number of respondents
  K = 3, # number of latent attributes
  J = 21, # number of items
  indT = 3, # number of time points
  N_dataset = 1, # number of datasets to generate
  seed = 2025, # seed for reproducibility
  Q_mat = Q_mat # generate a random Q-matrix if NULL
)

res <- HMLCDM_VB(
  data = data,
  max_iter = 100, # maximum number of iterations
  elbo = TRUE
)

df <- as.data.frame(res$alpha_trace)

# (2) Use base::stack() to reshape into “long” form
#    stack(df) creates a two‐column data.frame: values and ind (the original column name)
long_df <- stack(df)
#    Add an iteration index for each row of each original column:
long_df$iter <- rep(1:nrow(df), times = ncol(df))
#    Rename columns for ggplot
names(long_df) <- c("value", "variable", "iter")

# (3) Plot with ggplot2
ggplot(long_df, aes(x = iter, y = value, color = variable)) +
  geom_line() +
  theme_minimal() +
  labs(x = "Iteration", y = "Value", color = "Trajectory")

