#!/usr/bin/env Rscript
# APPNN Amyloid Prediction Analysis Script
# 
# This script provides comprehensive APPNN prediction capabilities:
# - Batch FASTA processing
# - Per-residue score extraction  
# - Hotspot region detection
# - Publication-quality visualizations
# - TSV export matching AmyloidBench format
#
# Reference:
# Família C, et al. (2015) PLoS ONE 10(8):e0134355
# DOI: 10.1371/journal.pone.0134355
#
# Usage:
#   Rscript APPNN_analysis.R input.fasta output_directory [threshold]
#
# Or source and use interactively:
#   source("APPNN_analysis.R")
#   results <- analyze_fasta("proteins.fasta", "results/")

# =============================================================================
# DEPENDENCIES
# =============================================================================

# Install required packages if not present
required_packages <- c("appnn", "ggplot2", "dplyr", "stringr", "readr", "purrr", "tibble")

for (pkg in required_packages) {
  if (!require(pkg, quietly = TRUE, character.only = TRUE)) {
    message(sprintf("Installing package: %s", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org/", quiet = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# Null coalescing operator
`%||%` <- function(a, b) if (!is.null(a)) a else b

# =============================================================================
# FASTA PARSING
# =============================================================================

#' Read FASTA file into named list
#'
#' @param fasta_file Path to FASTA file
#' @return Named list with sequence IDs as names and sequences as values
read_fasta_to_list <- function(fasta_file) {
  if (!file.exists(fasta_file)) {
    stop(sprintf("FASTA file not found: %s", fasta_file))
  }
  
  lines <- readLines(fasta_file)
  fasta_data <- list()
  current_header <- NULL
  current_sequence <- ""
  
  for (line in lines) {
    line <- trimws(line)
    if (nchar(line) == 0) next
    
    if (startsWith(line, ">")) {
      if (!is.null(current_header)) {
        fasta_data[[current_header]] <- current_sequence
      }
      # Parse header - remove '>' and take first word as ID
      header_text <- sub("^>", "", line)
      current_header <- strsplit(header_text, "\\s+")[[1]][1]
      current_sequence <- ""
    } else {
      current_sequence <- paste0(current_sequence, toupper(gsub("[^A-Za-z]", "", line)))
    }
  }
  
  if (!is.null(current_header) && nchar(current_sequence) > 0) {
    fasta_data[[current_header]] <- current_sequence
  }
  
  message(sprintf("Loaded %d sequences from %s", length(fasta_data), fasta_file))
  return(fasta_data)
}

# =============================================================================
# HOTSPOT PARSING
# =============================================================================

#' Parse hotspots from APPNN output into consistent format
#'
#' @param hs Raw hotspot data from APPNN
#' @param index_base Index base (0 or 1)
#' @return Tibble with start/end columns
parse_hotspots <- function(hs, index_base = 1L) {
  acc <- list()
  
  add_pair <- function(a, b) {
    a <- suppressWarnings(as.integer(a))
    b <- suppressWarnings(as.integer(b))
    if (is.na(a) || is.na(b)) return(invisible(NULL))
    s <- min(a, b)
    e <- max(a, b)
    if (index_base == 0L) {
      s <- s + 1L
      e <- e + 1L
    }
    acc[[length(acc) + 1L]] <<- c(s, e)
  }
  
  walk_any <- function(x) {
    if (is.null(x) || (length(x) == 0)) return(invisible(NULL))
    
    # data.frame with start/end
    if (is.data.frame(x) && all(c("start", "end") %in% names(x))) {
      for (i in seq_len(nrow(x))) add_pair(x$start[i], x$end[i])
      return(invisible(NULL))
    }
    
    # matrix with >= 2 columns
    if (is.matrix(x) && ncol(x) >= 2) {
      for (i in seq_len(nrow(x))) add_pair(x[i, 1], x[i, 2])
      return(invisible(NULL))
    }
    
    # atomic numeric vector
    if (is.atomic(x) && is.numeric(x)) {
      if (length(x) >= 2L) add_pair(x[1], x[2])
      return(invisible(NULL))
    }
    
    # atomic character: extract "start:end" or "start-end"
    if (is.atomic(x) && is.character(x)) {
      pats <- stringr::str_match_all(x, "(\\d+)\\s*[:\\-]\\s*(\\d+)")
      for (m in pats) {
        if (!is.null(m) && nrow(m)) {
          for (i in seq_len(nrow(m))) add_pair(m[i, 2], m[i, 3])
        }
      }
      return(invisible(NULL))
    }
    
    # list: recurse
    if (is.list(x)) {
      for (el in x) walk_any(el)
      return(invisible(NULL))
    }
    
    invisible(NULL)
  }
  
  walk_any(hs)
  
  if (!length(acc)) {
    return(tibble::tibble(start = integer(), end = integer()))
  }
  
  out <- tibble::tibble(
    start = vapply(acc, `[[`, integer(1), 1),
    end = vapply(acc, `[[`, integer(1), 2)
  ) |>
    dplyr::filter(!is.na(start) & !is.na(end)) |>
    dplyr::mutate(
      start = pmin(start, end),
      end = pmax(start, end)
    ) |>
    dplyr::distinct() |>
    dplyr::arrange(start, end)
  
  return(out)
}

# =============================================================================
# RESULT PROCESSING
# =============================================================================

#' Tidy a single APPNN result entry
#'
#' @param entry Single APPNN result entry
#' @param threshold Score threshold for hotspot classification
#' @return List with tbl (tibble), hs (hotspots), id, overall
tidy_one_appnn <- function(entry, threshold = 0.5) {
  seq_str <- entry$sequence
  aa_vec <- strsplit(seq_str, "", fixed = TRUE)[[1]]
  n_seq <- length(aa_vec)
  scores <- entry$aminoacids
  
  # Handle length mismatch
  if (length(scores) != n_seq) {
    N <- min(n_seq, length(scores))
    aa_vec <- aa_vec[seq_len(N)]
    scores <- scores[seq_len(N)]
    n_seq <- N
    warning(sprintf("Length mismatch for '%s': truncated to %d residues.",
                    entry$id %||% "sequence", N))
  }
  
  # Parse hotspots
  hs_raw <- if (!is.null(entry$hotspots)) entry$hotspots else entry$hotspot
  hs_df <- parse_hotspots(hs_raw, index_base = 1L)
  
  # Clamp hotspot bounds to sequence length
  if (nrow(hs_df)) {
    hs_df$start <- pmax(1L, pmin(hs_df$start, n_seq))
    hs_df$end <- pmax(1L, pmin(hs_df$end, n_seq))
  }
  
  # Build residue-level hotspot flags
  is_hs <- rep(FALSE, n_seq)
  if (nrow(hs_df)) {
    for (i in seq_len(nrow(hs_df))) {
      is_hs[hs_df$start[i]:hs_df$end[i]] <- TRUE
    }
  }
  
  # If no hotspots from APPNN, detect from threshold
  if (!any(is_hs) && any(scores >= threshold)) {
    is_hs <- scores >= threshold
    # Extract regions
    in_region <- FALSE
    region_start <- 0
    new_regions <- list()
    
    for (i in seq_along(is_hs)) {
      if (is_hs[i]) {
        if (!in_region) {
          region_start <- i
          in_region <- TRUE
        }
      } else {
        if (in_region) {
          if ((i - 1 - region_start + 1) >= 4) {  # Min length 4
            new_regions[[length(new_regions) + 1]] <- c(region_start, i - 1)
          }
          in_region <- FALSE
        }
      }
    }
    if (in_region && (n_seq - region_start + 1) >= 4) {
      new_regions[[length(new_regions) + 1]] <- c(region_start, n_seq)
    }
    
    if (length(new_regions) > 0) {
      hs_df <- tibble::tibble(
        start = sapply(new_regions, `[[`, 1),
        end = sapply(new_regions, `[[`, 2)
      )
    }
  }
  
  # Normalize scores to 0-1 range
  score_min <- min(scores, na.rm = TRUE)
  score_max <- max(scores, na.rm = TRUE)
  if (score_max > score_min) {
    normalized <- (scores - score_min) / (score_max - score_min)
  } else {
    normalized <- rep(0.5, n_seq)
  }
  
  tbl <- tibble::tibble(
    id = entry$id %||% attr(entry, "id") %||% "sequence",
    pos = seq_len(n_seq),
    aa = aa_vec,
    score = as.numeric(scores),
    normalized = normalized,
    is_hotspot = is_hs,
    overall = as.numeric(entry$overall %||% NA)
  )
  
  list(
    tbl = tbl,
    hs = hs_df,
    id = entry$id %||% attr(entry, "id") %||% "sequence",
    overall = as.numeric(entry$overall %||% NA)
  )
}

# =============================================================================
# VISUALIZATION
# =============================================================================

#' Plot APPNN results for a single sequence
#'
#' @param appnn_entry Single APPNN result entry
#' @param threshold Score threshold (for display)
#' @param tick_step Major tick interval
#' @param minor_step Minor tick interval
#' @param band_pad Padding for hotspot bands
#' @return ggplot object
plot_appnn_sequence <- function(appnn_entry,
                                 threshold = 0.5,
                                 tick_step = 10,
                                 minor_step = 5,
                                 band_pad = 0.15) {
  prepared <- tidy_one_appnn(appnn_entry, threshold = threshold)
  df <- prepared$tbl
  hsd <- prepared$hs
  
  title_txt <- prepared$id
  if (!is.na(prepared$overall)) {
    title_txt <- paste0(title_txt, "  (overall=", round(prepared$overall, 3), ")")
  }
  
  xmax <- max(df$pos)
  
  p <- ggplot(df, aes(x = pos, y = score)) +
    # Hotspot bands
    {
      if (nrow(hsd)) {
        geom_rect(
          data = transform(hsd,
                           xmin = start - band_pad,
                           xmax = end + band_pad),
          inherit.aes = FALSE,
          aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
          fill = "#D55E00",
          alpha = 0.15
        )
      }
    } +
    # Threshold line
    geom_hline(yintercept = threshold, linetype = "dashed", color = "#888888") +
    # Score line
    geom_line(color = "#333333") +
    # Points colored by hotspot status
    geom_point(aes(color = is_hotspot), size = 1.7) +
    scale_color_manual(
      values = c(`TRUE` = "#D55E00", `FALSE` = "#000000"),
      name = "Amyloidogenic",
      labels = c(`TRUE` = "Yes", `FALSE` = "No")
    ) +
    scale_x_continuous(
      breaks = seq(0, xmax, by = tick_step),
      minor_breaks = if (!is.null(minor_step) && minor_step > 0) seq(0, xmax, by = minor_step) else NULL,
      expand = expansion(mult = c(0, 0.01))
    ) +
    scale_y_continuous(
      limits = c(0, 1),
      expand = expansion(mult = c(0, 0.02))
    ) +
    labs(
      title = title_txt,
      x = "Residue Position",
      y = "APPNN Propensity Score"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold"),
      legend.position = "top",
      panel.grid.minor.x = element_line(linewidth = 0.2),
      panel.grid.major.y = element_line(linewidth = 0.3, color = "#CCCCCC")
    )
  
  return(p)
}

#' Plot multiple sequences in a faceted layout
#'
#' @param appnn_results List of APPNN results
#' @param threshold Score threshold
#' @param ncol Number of columns in facet grid
#' @return ggplot object
plot_appnn_multi <- function(appnn_results, threshold = 0.5, ncol = 2) {
  # Combine all data
  all_data <- purrr::map_dfr(appnn_results, function(entry) {
    tidy_one_appnn(entry, threshold = threshold)$tbl
  })
  
  # Get hotspot data for all
  all_hs <- purrr::map_dfr(appnn_results, function(entry) {
    prepared <- tidy_one_appnn(entry, threshold = threshold)
    if (nrow(prepared$hs) > 0) {
      prepared$hs$id <- prepared$id
      return(prepared$hs)
    }
    return(tibble::tibble())
  })
  
  p <- ggplot(all_data, aes(x = pos, y = score)) +
    {
      if (nrow(all_hs) > 0) {
        geom_rect(
          data = transform(all_hs, xmin = start - 0.15, xmax = end + 0.15),
          inherit.aes = FALSE,
          aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
          fill = "#D55E00",
          alpha = 0.15
        )
      }
    } +
    geom_hline(yintercept = threshold, linetype = "dashed", color = "#888888") +
    geom_line(color = "#333333") +
    geom_point(aes(color = is_hotspot), size = 1) +
    scale_color_manual(
      values = c(`TRUE` = "#D55E00", `FALSE` = "#000000"),
      name = "Amyloidogenic"
    ) +
    facet_wrap(~id, scales = "free_x", ncol = ncol) +
    labs(x = "Position", y = "APPNN Score") +
    theme_minimal(base_size = 10) +
    theme(
      legend.position = "top",
      strip.text = element_text(face = "bold")
    )
  
  return(p)
}

# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

#' Export APPNN results to TSV and plots
#'
#' @param appnn_obj APPNN result object (single or list)
#' @param out_dir Output directory
#' @param save_plots Save plot PNG files
#' @param threshold Score threshold
#' @param width Plot width in inches
#' @param height Plot height in inches
#' @param dpi Plot resolution
#' @return List of export results (invisible)
export_appnn <- function(appnn_obj,
                          out_dir = "appnn_exports",
                          save_plots = TRUE,
                          threshold = 0.5,
                          width = 10,
                          height = 4,
                          dpi = 300) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Coerce to list of entries
  entries <- if (inherits(appnn_obj, "appnn")) {
    unclass(appnn_obj)
  } else if (is.list(appnn_obj) && !is.null(appnn_obj$sequence)) {
    list(appnn_obj)  # Single entry
  } else if (is.list(appnn_obj)) {
    appnn_obj
  } else {
    stop("`appnn_obj` must be an S3 'appnn' object or a list of entries.")
  }
  
  results <- purrr::map(entries, function(entry) {
    prepared <- tidy_one_appnn(entry, threshold = threshold)
    df <- prepared$tbl
    hs_df <- prepared$hs
    
    # Safe filename
    id <- prepared$id
    id_safe <- stringr::str_replace_all(id, "[^A-Za-z0-9._-]", "_")
    
    # Save per-residue TSV
    tsv_path <- file.path(out_dir, paste0(id_safe, ".tsv"))
    readr::write_tsv(df, tsv_path)
    
    # Save regions TSV if any
    regions_path <- NA_character_
    if (nrow(hs_df) > 0) {
      regions_df <- hs_df |>
        dplyr::mutate(
          id = id,
          length = end - start + 1,
          sequence = purrr::map2_chr(start, end, function(s, e) {
            paste(df$aa[s:e], collapse = "")
          })
        ) |>
        dplyr::select(id, start, end, length, sequence)
      
      regions_path <- file.path(out_dir, paste0(id_safe, "_regions.tsv"))
      readr::write_tsv(regions_df, regions_path)
    }
    
    # Save plot
    png_path <- NA_character_
    if (isTRUE(save_plots)) {
      p <- plot_appnn_sequence(entry, threshold = threshold)
      png_path <- file.path(out_dir, paste0(id_safe, ".png"))
      ggplot2::ggsave(
        filename = png_path,
        plot = p,
        width = width,
        height = height,
        dpi = dpi,
        limitsize = FALSE
      )
    }
    
    list(id = id, tsv = tsv_path, regions = regions_path, plot = png_path)
  })
  
  # Summary TSV
  summary_df <- purrr::map_dfr(entries, function(entry) {
    prepared <- tidy_one_appnn(entry, threshold = threshold)
    tibble::tibble(
      id = prepared$id,
      length = nrow(prepared$tbl),
      overall_score = prepared$overall,
      n_regions = nrow(prepared$hs),
      n_hotspot_residues = sum(prepared$tbl$is_hotspot),
      hotspot_fraction = mean(prepared$tbl$is_hotspot)
    )
  })
  readr::write_tsv(summary_df, file.path(out_dir, "summary.tsv"))
  
  message(sprintf("Exported %d sequences to %s", length(results), out_dir))
  invisible(results)
}

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

#' Analyze sequences from FASTA file using APPNN
#'
#' @param fasta_file Path to input FASTA file
#' @param output_dir Output directory for results
#' @param threshold Amyloidogenicity threshold (default 0.5)
#' @param save_plots Generate visualization plots
#' @return APPNN result object
analyze_fasta <- function(fasta_file,
                           output_dir = "appnn_results",
                           threshold = 0.5,
                           save_plots = TRUE) {
  # Read sequences
  sequences <- read_fasta_to_list(fasta_file)
  
  if (length(sequences) == 0) {
    stop("No valid sequences found in FASTA file")
  }
  
  # Filter sequences >= 6 aa
  sequences <- sequences[sapply(sequences, nchar) >= 6]
  message(sprintf("Processing %d sequences (length >= 6)", length(sequences)))
  
  # Run APPNN
  message("Running APPNN predictions...")
  appnn_results <- appnn::appnn(unname(sequences))
  
  # Attach IDs
  names_vec <- names(sequences)
  for (i in seq_along(appnn_results)) {
    appnn_results[[i]]$id <- names_vec[i]
  }
  
  # Export results
  export_appnn(
    appnn_results,
    out_dir = output_dir,
    save_plots = save_plots,
    threshold = threshold
  )
  
  return(appnn_results)
}

#' Analyze a single sequence
#'
#' @param sequence Protein sequence string
#' @param sequence_id Sequence identifier
#' @param output_dir Output directory
#' @param threshold Amyloidogenicity threshold
#' @param save_plots Generate plot
#' @return APPNN result
analyze_sequence <- function(sequence,
                              sequence_id = "query",
                              output_dir = "appnn_results",
                              threshold = 0.5,
                              save_plots = TRUE) {
  if (nchar(sequence) < 6) {
    stop("Sequence must be at least 6 amino acids long")
  }
  
  # Clean sequence
  sequence <- toupper(gsub("[^A-Za-z]", "", sequence))
  
  # Run APPNN
  appnn_results <- appnn::appnn(sequence)
  appnn_results[[1]]$id <- sequence_id
  
  # Export
  export_appnn(
    appnn_results,
    out_dir = output_dir,
    save_plots = save_plots,
    threshold = threshold
  )
  
  return(appnn_results[[1]])
}

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) < 2) {
    cat("APPNN Amyloid Prediction Analysis\n")
    cat("Usage: Rscript APPNN_analysis.R input.fasta output_directory [threshold]\n")
    cat("\n")
    cat("Arguments:\n")
    cat("  input.fasta      - Input FASTA file with protein sequences\n")
    cat("  output_directory - Directory for output files\n")
    cat("  threshold        - Amyloidogenicity threshold (default: 0.5)\n")
    cat("\n")
    cat("Output files:\n")
    cat("  <id>.tsv         - Per-residue scores for each sequence\n")
    cat("  <id>_regions.tsv - Detected amyloidogenic regions\n")
    cat("  <id>.png         - Visualization plot\n")
    cat("  summary.tsv      - Summary statistics for all sequences\n")
    quit(status = 1)
  }
  
  input_fasta <- args[1]
  output_dir <- args[2]
  threshold <- if (length(args) >= 3) as.numeric(args[3]) else 0.5
  
  cat(sprintf("Input:     %s\n", input_fasta))
  cat(sprintf("Output:    %s\n", output_dir))
  cat(sprintf("Threshold: %.2f\n", threshold))
  cat("\n")
  
  tryCatch({
    results <- analyze_fasta(
      fasta_file = input_fasta,
      output_dir = output_dir,
      threshold = threshold,
      save_plots = TRUE
    )
    cat(sprintf("\nSuccessfully processed %d sequences\n", length(results)))
  }, error = function(e) {
    cat(sprintf("Error: %s\n", e$message))
    quit(status = 1)
  })
}
