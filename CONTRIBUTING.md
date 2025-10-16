# Contributing to HM-LACDM

Thank you for your interest in contributing to HM-LACDM! This document provides guidelines for contributing to the package.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:

1. A clear, descriptive title
2. Steps to reproduce the bug
3. Expected vs. actual behavior
4. Your R version and package version
5. A minimal reproducible example (reprex)

Example:

```r
# Minimal reproducible example
library(HM-LACDM)
data <- data_generate(i = 10, k = 2, j = 5, t = 2)
results <- hmlcdm_vb(data)  # Error occurs here
```

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue with:

1. Clear description of the proposed feature
2. Use case explaining why it would be useful
3. Example of how it would work (if applicable)

### Pull Requests

We welcome pull requests! Please follow these steps:

1. **Fork the repository** and create a new branch from `main`
2. **Make your changes**:
   - Follow the existing code style
   - Add roxygen2 documentation for new functions
   - Add examples to function documentation
   - Update NEWS.md with your changes
3. **Test your changes**:
   - Ensure all existing tests pass
   - Add new tests for new functionality
   - Run `devtools::check()` locally
4. **Submit the pull request** with:
   - Clear description of changes
   - Reference to related issues (if any)

## Development Setup

### Prerequisites

- R >= 4.1.0
- RStudio (recommended)
- Git

### Getting Started

```r
# Clone your fork
git clone https://github.com/YOUR_USERNAME/HMLCDM.git
cd HMLCDM

# Install development dependencies
install.packages(c("devtools", "roxygen2", "testthat", "knitr"))

# Install package dependencies
devtools::install_deps()

# Load the package for development
devtools::load_all()
```

### Development Workflow

```r
# Make changes to code

# Update documentation
devtools::document()

# Run tests
devtools::test()

# Check package
devtools::check()

# Build and install
devtools::install()
```

## Code Style

- Use meaningful variable and function names
- Follow tidyverse style guide where applicable
- Add comments for complex logic
- Keep functions focused and modular
- Use roxygen2 for documentation

### Function Documentation Template

```r
#' Brief Title
#'
#' Longer description explaining what the function does.
#'
#' @param param1 Description of parameter 1
#' @param param2 Description of parameter 2
#'
#' @return Description of what the function returns
#'
#' @examples
#' \dontrun{
#' # Example usage
#' result <- my_function(param1 = value1, param2 = value2)
#' }
#'
#' @export
my_function <- function(param1, param2) {
  # Function implementation
}
```

## Testing

- Add tests for new functions
- Ensure tests cover edge cases
- Tests should be in `tests/testthat/`
- Use descriptive test names

Example test structure:

```r
test_that("function_name works correctly", {
  # Setup
  input <- create_test_input()

  # Execute
  result <- my_function(input)

  # Assert
  expect_equal(result$value, expected_value)
  expect_true(is.numeric(result$value))
})
```

## Documentation

- Update README.md for user-facing changes
- Update NEWS.md for all changes
- Add or update vignettes for major features
- Ensure all exported functions have complete documentation

## Questions?

If you have questions about contributing, feel free to:

- Open an issue with the "question" label
- Email the maintainer: hduan7@ucla.edu

## Code of Conduct

Please note that this project follows a standard Code of Conduct:

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good faith

Thank you for contributing to HM-LACDM!
