# Multi-run HMLCDM with best model selection based on profile accuracy
#
# This function runs the HMLCDM algorithm multiple times and selects
# the best result based on profile accuracy and other criteria

#' Run HMLCDM Multiple Times and Select Best Result
#'
#' @param data Input data list containing y, k, and ground_truth
#' @param n_runs Number of independent runs (default: 3)
#' @param max_iter Maximum iterations per run (default: 100)
#' @param alpha_level Significance level for Q-matrix recovery (default: 0.05)
#' @param min_profile_threshold Minimum acceptable mean profile accuracy,
#' (default: 0.6)
#' @param verbose Whether to print progress (default: TRUE)
#' @param seed_offset Starting seed offset for reproducibility (default: 1000)
#'
#' @return List containing:
#'   - best_result: The best model result
#'   - all_results: All model results for comparison
#'   - selection_metrics: Metrics used for selection
#'   - best_run_index: Index of the best run
#'
#' @export
multi_run_hmlcdm <- function(
    data, n_runs = 3, max_iter = 100, alpha_level = 0.05,
    min_profile_threshold = 0.6, verbose = TRUE, seed_offset = 1000) {
  if (verbose) {
    cat("Running HMLCDM", n_runs, "times with profile accuracy selection...\n")
    cat("Minimum profile accuracy threshold:", min_profile_threshold, "\n\n")
  }
  # Storage for results
  all_results <- vector("list", n_runs)
  selection_metrics <- data.frame(
    run = 1:n_runs,
    final_elbo = numeric(n_runs),
    q_accuracy = numeric(n_runs),
    mean_profile_acc = numeric(n_runs),
    min_profile_acc = numeric(n_runs),
    runtime = numeric(n_runs),
    converged = logical(n_runs),
    stringsAsFactors = FALSE
  )

  # Run algorithm multiple times
  for (i in 1:n_runs) {
    if (verbose) {
      cat("Run", i, "of", n_runs, "...")
    }

    start_time <- Sys.time()

    # Set different seed for each run
    torch_manual_seed(seed_offset + i) # nolint
    set.seed(seed_offset + i)

    # Run the algorithm
    result <- tryCatch(
      {
        hmlcdm_vb( # nolint
          data = data,
          max_iter = max_iter,
          elbo = TRUE, # Always track ELBO for selection
          alpha_level = alpha_level
        )
      },
      error = function(e) {
        if (verbose) cat(" ERROR:", e$message, "\n")
        return(NULL)
      }
    )

    end_time <- Sys.time()
    runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))

    if (!is.null(result)) {
      all_results[[i]] <- result

      # Extract metrics for selection
      final_elbo <- tail(result$elbo, 1)
      q_accuracy <- if (!is.null(result$Q_acc)) result$Q_acc else 0

      # Profile accuracy metrics
      if (!is.null(result$profiles_acc)) {
        mean_profile_acc <- mean(result$profiles_acc)
        min_profile_acc <- min(result$profiles_acc)
      } else {
        mean_profile_acc <- 0
        min_profile_acc <- 0
      }

      # Check convergence (ELBO change in last 10 iterations)
      if (length(result$elbo) >= 10) {
        elbo_tail <- tail(result$elbo, 10)
        elbo_change <- abs(elbo_tail[10] - elbo_tail[1]) / abs(elbo_tail[1])
        converged <- elbo_change < 0.001 # Less than 0.1% change
      } else {
        converged <- FALSE
      }

      selection_metrics[i, "run"] <- i
      selection_metrics[i, "final_elbo"] <- final_elbo
      selection_metrics[i, "q_accuracy"] <- q_accuracy
      selection_metrics[i, "mean_profile_acc"] <- mean_profile_acc
      selection_metrics[i, "min_profile_acc"] <- min_profile_acc
      selection_metrics[i, "runtime"] <- runtime
      selection_metrics[i, "converged"] <- converged

      if (verbose) {
        cat(
          " ELBO:", round(final_elbo, 1),
          "| Q-acc:", round(q_accuracy, 3),
          "| Profile-acc:", round(mean_profile_acc, 3),
          "| Min-profile:", round(min_profile_acc, 3),
          "| Time:", round(runtime, 1), "s",
          if (converged) " ✓" else " ✗", "\n"
        )
      }
    } else {
      selection_metrics[i, "run"] <- i
      selection_metrics[i, "final_elbo"] <- -Inf
      selection_metrics[i, "q_accuracy"] <- 0
      selection_metrics[i, "mean_profile_acc"] <- 0
      selection_metrics[i, "min_profile_acc"] <- 0
      selection_metrics[i, "runtime"] <- runtime
      selection_metrics[i, "converged"] <- FALSE
      if (verbose) cat(" FAILED\n")
    }
  }

  # Remove failed runs
  successful_runs <- which(!is.na(selection_metrics$final_elbo) &
                             is.finite(selection_metrics$final_elbo))

  if (length(successful_runs) == 0) {
    stop("All runs failed!")
  }

  # Select best run
  best_run_index <- select_best_run(
    selection_metrics[successful_runs, ],
    min_profile_threshold, verbose
  )

  actual_best_index <- successful_runs[best_run_index]

  if (verbose) {
    cat("\n=== SELECTION SUMMARY ===\n")
    cat("Best run:", actual_best_index, "\n")
    cat("Profile accuracy:",
        round(selection_metrics[actual_best_index, "mean_profile_acc"], 3),
        "\n")
    cat("Q-matrix accuracy:",
        round(selection_metrics[actual_best_index, "q_accuracy"], 3), "\n")
    cat("Final ELBO:",
        round(selection_metrics[actual_best_index, "final_elbo"], 1), "\n")
    cat("Converged:", selection_metrics[actual_best_index, "converged"], "\n")
  }

  return(list(
    best_result = all_results[[actual_best_index]],
    all_results = all_results[successful_runs],
    selection_metrics = selection_metrics[successful_runs, ],
    best_run_index = actual_best_index,
    n_successful = length(successful_runs)
  ))
}

#' Select Best Run Based on Profile Accuracy and Other Criteria
#'
#' @param metrics Data frame with selection metrics
#' @param min_profile_threshold Minimum acceptable profile accuracy
#' @param verbose Whether to print selection details
#'
#' @return Index of best run
select_best_run <- function(metrics, min_profile_threshold, verbose = TRUE) {

  # Priority 1: Runs with good profile accuracy
  good_profile_runs <- which(metrics$mean_profile_acc >= min_profile_threshold)

  if (length(good_profile_runs) > 0) {
    if (verbose) {
      cat(
        "Found", length(good_profile_runs), "runs with profile accuracy >=",
        min_profile_threshold, "\n"
      )
    }

    # Among good profile runs, prefer converged ones
    good_converged <- intersect(good_profile_runs, which(metrics$converged))

    if (length(good_converged) > 0) {
      if (verbose) cat("Selecting from", length(good_converged),
                       "converged runs with good profiles\n")
      # Among good converged runs, pick highest ELBO
      candidate_metrics <- metrics[good_converged, ]
      best_among_candidates <- which.max(candidate_metrics$final_elbo)
      return(good_converged[best_among_candidates])
    } else {
      if (verbose) {
        cat("No converged runs with good profiles, 
        selecting best ELBO among good profiles\n")
      }
      # No converged good runs, pick best ELBO among good profile runs
      candidate_metrics <- metrics[good_profile_runs, ]
      best_among_candidates <- which.max(candidate_metrics$final_elbo)
      return(good_profile_runs[best_among_candidates])
    }
  } else {
    if (verbose) {
      cat(
        "No runs with profile accuracy >=", min_profile_threshold,
        ", selecting best available\n"
      )
    }

    # No runs meet threshold, pick the one with highest profile accuracy
    # (this handles the case where all runs have poor profile accuracy)
    best_profile_idx <- which.max(metrics$mean_profile_acc)

    if (verbose) {
      cat(
        "Best available profile accuracy:",
        round(metrics$mean_profile_acc[best_profile_idx], 3), "\n"
      )
    }

    return(best_profile_idx)
  }
}

#' Compare Multi-Run Results
#'
#' @param multi_run_result Result from multi_run_hmlcdm
#'
#' @export
compare_runs <- function(multi_run_result) {
  metrics <- multi_run_result$selection_metrics

  cat("=== MULTI-RUN COMPARISON ===\n")
  cat("Total successful runs:", multi_run_result$n_successful, "\n")
  cat("Best run index:", multi_run_result$best_run_index, "\n\n")

  cat("Profile Accuracy Summary:\n")
  cat("  Mean:", round(mean(metrics$mean_profile_acc), 3), "\n")
  cat(
    "  Range: [", round(min(metrics$mean_profile_acc), 3), ", ",
    round(max(metrics$mean_profile_acc), 3), "]\n"
  )
  cat("  Runs > 0.8:", sum(metrics$mean_profile_acc > 0.8), "\n")
  cat("  Runs > 0.6:", sum(metrics$mean_profile_acc > 0.6), "\n")
  cat("  Runs < 0.4:", sum(metrics$mean_profile_acc < 0.4), "\n\n")

  cat("Other Metrics Summary:\n")
  print(summary(metrics[, c("final_elbo", "q_accuracy", "runtime")]))

  cat(
    "\nConvergence rate:",
    round(mean(metrics$converged) * 100, 1), "%\n"
  )

  cat("\nTop 3 runs by profile accuracy:\n")
  top_profile <- head(metrics[order(metrics$mean_profile_acc,
                                    decreasing = TRUE), ], 3)
  print(top_profile[, c("run", "mean_profile_acc",
                        "q_accuracy", "final_elbo", "converged")])

  invisible(metrics)
}