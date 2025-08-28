#' Check all R and Python dependencies for the DNA annotation pipeline
#'
#' This script checks R packages, rDNA installation, Conda, the conda environment,
#' and all required Python packages + versions. It does not install anything.
#' It prints all missing dependencies clearly.
#'
#' @param env_name Name of the Conda environment to check (default: "dna-pipeline")
#' @export
check_pipeline_dependencies <- function(env_name = "dna-pipeline") {
  cat("🔎 Checking R packages...\n")
  required_r_packages <- c("reticulate", "rJava", "remotes")
  missing_r <- required_r_packages[!vapply(required_r_packages, requireNamespace, logical(1), quietly = TRUE)]

  if (length(missing_r) > 0) {
    cat("❌ Missing R packages: ", paste(missing_r, collapse = ", "), "\n")
  } else {
    cat("✅ All required R packages are installed.\n")
  }

  # Check if rDNA is installed
  cat("\n🔎 Checking rDNA package...\n")
  if (!requireNamespace("rDNA", quietly = TRUE)) {
    cat("❌ rDNA is not installed. Please install it with:\n")
    cat('   remotes::install_github("leifeld/dna/rDNA/rDNA@*release")\n')
  } else {
    cat("✅ rDNA is installed.\n")
  }

  # Check for Conda
  cat("\n🔎 Checking for Conda installation...\n")
  conda_path <- reticulate::conda_binary()
  if (is.null(conda_path) || conda_path == "") {
    cat("❌ Conda is not installed or not found in your PATH.\n")
    cat("   Please install Anaconda (https://www.anaconda.com/) or Miniconda (https://docs.conda.io/en/latest/miniconda.html)\n")
    return(invisible(FALSE))
  } else {
    cat("✅ Conda is available at: ", conda_path, "\n")
  }

  # Check for environment
  cat("\n🔎 Checking for Conda environment: ", env_name, "\n")
  envs <- reticulate::conda_list()$name
  if (!(env_name %in% envs)) {
    cat("❌ Conda environment '", env_name, "' does not exist.\n", sep = "")
    cat("   You can create it using setup_pipeline() or conda CLI.\n")
    return(invisible(FALSE))
  } else {
    cat("✅ Conda environment '", env_name, "' exists.\n", sep = "")
  }

  # Use the environment
  reticulate::use_condaenv(env_name, required = TRUE)

  # Python package requirements (min versions)
  cat("\n🔎 Checking required Python packages and versions...\n")
  requirements <- list(
    pandas = "2.0.0",
    numpy = "1.24.0",
    sklearn = "1.3.0",
    joblib = "1.3.0",
    transformers = "4.40.0",
    sentence_transformers = "2.6.1",
    spacy = "3.7.4"
  )

  issues <- list()

  for (pkg in names(requirements)) {
    if (!reticulate::py_module_available(pkg)) {
      issues[[pkg]] <- "❌ Not installed"
    } else {
      version <- tryCatch({
        reticulate::py_run_string(sprintf("import %s; __ver = %s.__version__", pkg, pkg))$`__ver`
      }, error = function(e) NA)

      if (is.na(version)) {
        issues[[pkg]] <- "⚠️ Installed but version check failed"
      } else if (utils::compareVersion(version, requirements[[pkg]]) < 0) {
        issues[[pkg]] <- paste0("❌ Version ", version, " < required ", requirements[[pkg]])
      } else {
        issues[[pkg]] <- paste0("✅ Version ", version)
      }
    }
  }

  # Print results
  for (pkg in names(issues)) {
    cat(sprintf("%-22s : %s\n", pkg, issues[[pkg]]))
  }

  # Final summary
  if (any(grepl("❌", unlist(issues)))) {
    cat("\n⚠️ Some Python dependencies are missing or outdated.\n")
    cat("   You can fix them by running: setup_pipeline()\n")
    return(invisible(FALSE))
  } else {
    cat("\n✅ All Python dependencies are installed and meet version requirements.\n")
    cat("🎉 No setup needed. You’re ready to run the pipeline!\n")
    cat("   Use: run_pipeline(\"your_file.txt\")\n")
    return(invisible(TRUE))
  }
}
