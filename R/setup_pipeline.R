#' Set up the DNA pipeline environment (R packages, Conda, Python packages)
#'
#' This function installs all required R packages, checks or installs Conda (Miniconda),
#' creates a Conda environment named "dna-pipeline" if it does not exist, and installs
#' all required Python packages with version requirements. Should be run once by new users.
#'
#' @param env_name Name of the Conda environment to create/use (default = "dna-pipeline")
#' @param install_miniconda If TRUE, installs Miniconda if Conda is missing (default = TRUE)
#' @export
setup_pipeline <- function(env_name = "dna-pipeline", install_miniconda = TRUE) {
  cat("🔧 Setting up the DNA annotation pipeline...\n\n")

  # --- 1. Install required R packages ---
  required_r_packages <- c("reticulate", "rJava", "remotes")
  cat("📦 Checking required R packages...\n")
  for (pkg in required_r_packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      cat("📥 Installing missing R package:", pkg, "\n")
      install.packages(pkg, repos = "https://cloud.r-project.org")
    } else {
      cat("✅ R package already installed:", pkg, "\n")
    }
  }

  # --- 2. Install rDNA from GitHub if missing ---
  if (!requireNamespace("rDNA", quietly = TRUE)) {
    cat("📥 Installing 'rDNA' from GitHub...\n")
    remotes::install_github("leifeld/dna/rDNA/rDNA@*release", INSTALL_opts = "--no-multiarch")
  } else {
    cat("✅ rDNA already installed.\n")
  }

  # --- 3. Check for Conda (Anaconda or Miniconda) ---
  cat("\n🐍 Checking Conda installation...\n")
  conda_path <- reticulate::conda_binary()

  if (is.null(conda_path) || conda_path == "") {
    if (install_miniconda) {
      cat("📥 Conda not found. Installing Miniconda...\n")
      reticulate::install_miniconda()
    } else {
      stop("❌ Conda is not available. Please install Anaconda or run with install_miniconda = TRUE.")
    }
  } else {
    cat("✅ Conda is available at:", conda_path, "\n")
  }

  # --- 4. Create the Conda environment if missing ---
  envs <- reticulate::conda_list()$name
  if (!(env_name %in% envs)) {
    cat("📦 Creating Conda environment '", env_name, "'...\n", sep = "")
    reticulate::conda_create(env_name, packages = "python=3.10")
  } else {
    cat("✅ Conda environment '", env_name, "' already exists.\n")
  }

  # --- 5. Bind to the environment safely ---
  cat("\n🔄 Activating Conda environment: ", env_name, "\n")
  reticulate::use_condaenv(env_name, required = TRUE)

  # --- 6. Define required Python packages + minimum versions ---
  requirements <- list(
    pandas = "2.0.0",
    numpy = "1.24.0",
    sklearn = "1.3.0",
    joblib = "1.3.0",
    transformers = "4.40.0",
    sentence_transformers = "2.6.1",
    spacy = "3.7.4"
  )

  # --- 7. Install or update Python packages ---
  cat("\n📦 Checking Python packages...\n")
  for (pkg in names(requirements)) {
    required_version <- requirements[[pkg]]
    module_available <- reticulate::py_module_available(pkg)

    if (module_available) {
      version <- tryCatch({
        reticulate::py_run_string(sprintf("import %s; __ver = %s.__version__", pkg, pkg))$`__ver`
      }, error = function(e) NA)

      if (is.na(version)) {
        cat("⚠️ Version check failed for", pkg, "- reinstalling...\n")
        reticulate::conda_install(env_name, pkg, pip = TRUE)
      } else if (utils::compareVersion(version, required_version) < 0) {
        cat("🔄 Updating", pkg, "from version", version, "to at least", required_version, "...\n")
        reticulate::conda_install(env_name, pkg, pip = TRUE)
      } else {
        cat("✅", pkg, "version", version, "meets requirement.\n")
      }

    } else {
      cat("📥 Installing missing Python package:", pkg, "\n")
      reticulate::conda_install(env_name, pkg, pip = TRUE)
    }
  }

  # --- 8. Download spacy model if needed ---
  cat("\n🧠 Ensuring spacy English model is available...\n")
  tryCatch({
    reticulate::py_run_string("import spacy; spacy.load('en_core_web_sm')")
    cat("✅ Spacy model 'en_core_web_sm' is already installed.\n")
  }, error = function(e) {
    cat("📥 Downloading spacy model 'en_core_web_sm'...\n")
    reticulate::py_run_string("import spacy; spacy.cli.download('en_core_web_sm')")
  })

  cat("\n✅ Setup complete! You can now run the pipeline with:\n")
  cat("   run_pipeline(\"your_file.txt\")\n")
}
