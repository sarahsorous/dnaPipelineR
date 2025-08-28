#' Run the full annotation pipeline
#'
#' This function runs all steps in the annotation pipeline:
#' actor/org extraction, statement segmentation, concept/right classification, sentiment analysis.
#' It prompts the user to choose a .txt file, and outputs a single merged CSV.
#'
#' @param env_name Name of the Conda environment to use (default: "dna-pipeline").
#' @export
run_pipeline <- function(env_name = "dna-pipeline") {
  # Load required R packages
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    install.packages("dplyr")
  }
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    install.packages("reticulate")
  }
  library(dplyr)

  # Prompt for file
  cat("📂 Please choose a .txt file to analyse...\n")
  file_path <- file.choose()
  if (!file.exists(file_path)) stop("❌ File not found: ", file_path)

  # Generate safe output name
  input_filename <- tools::file_path_sans_ext(basename(file_path))
  safe_name <- gsub("[^A-Za-z0-9_-]", "_", input_filename)
  final_output_path <- paste0("pipeline_output_", safe_name, ".csv")

  # Dependency check
  if (!check_pipeline_dependencies(env_name)) {
    stop("❌ Missing dependencies. Run setup_pipeline() first.")
  }

  # Activate Python environment
  reticulate::use_condaenv(env_name, required = TRUE)

  # Step 1: Actor/org extraction
  message("🔍 Step 1: Extracting actors and organisations...")
  reticulate::source_python(system.file("python/speaker_org_attribution_utils.py", package = "dnaPipelineR"))
  speaker_df <- extract_speaker_blocks_from_path(file_path)
  write.csv(speaker_df, "actor_organisation_output.csv", row.names = FALSE)

  # Step 2: Statement segmentation
  message("✂️ Step 2: Extracting statements...")
  reticulate::source_python(system.file("python/statement_segmenter_utils.py", package = "dnaPipelineR"))
  statement_segmenter_output_df <- extract_statements(
    input_csv = "actor_organisation_output.csv",
    output_csv = "actor_org_statements_confidence.csv",
    model_path = system.file("models/MLP_statement_segmenter.joblib", package = "dnaPipelineR")
  )

  # Step 3: Concept classification
  message("🧠 Step 3: Assigning concepts...")
  reticulate::source_python(system.file("python/concept_classifier_utils.py", package = "dnaPipelineR"))
  assign_concepts(
    input_csv = "actor_org_statements_confidence.csv",
    output_csv = "statement_concepts.csv",
    threshold = 0.6,
    model_path = system.file("models/concept_svm.joblib", package = "dnaPipelineR"),
    encoder_path = system.file("models/concept_label_encoder.joblib", package = "dnaPipelineR")
  )

  # Step 4: Right classification
  message("⚖️ Step 4: Assigning rights...")
  reticulate::source_python(system.file("python/right_classifier_utils.py", package = "dnaPipelineR"))
  assign_right_labels(
    input_csv = "actor_org_statements_confidence.csv",
    output_csv = "statement_rights.csv",
    model_path = system.file("models/right_svm.joblib", package = "dnaPipelineR"),
    encoder_path = system.file("models/right_label_encoder.joblib", package = "dnaPipelineR")
  )

  # Step 5: Sentiment classification
  message("💬 Step 5: Performing sentiment analysis...")
  reticulate::source_python(system.file("python/sentiment_analyser_utils.py", package = "dnaPipelineR"))
  sentiment_df <- analyse_sentiment(
    input_csv = "actor_org_statements_confidence.csv",
    output_csv = "actor_org_statements_with_sentiment.csv"
  )

  # Step 6: Merge outputs
  message("🧩 Merging results into final spreadsheet...")

  statements <- read.csv("actor_org_statements_confidence.csv", stringsAsFactors = FALSE)
  concepts <- read.csv("statement_concepts.csv", stringsAsFactors = FALSE)
  rights <- read.csv("statement_rights.csv", stringsAsFactors = FALSE)
  sentiment <- read.csv("actor_org_statements_with_sentiment.csv", stringsAsFactors = FALSE)

  final_df <- statements %>%
    left_join(concepts, by = c("speaker_org", "statement")) %>%
    left_join(rights, by = c("speaker_org", "statement")) %>%
    left_join(sentiment[, c("speaker_org", "statement", "sentiment_label")], by = c("speaker_org", "statement"))

  write.csv(final_df, final_output_path, row.names = FALSE)

  message("✅ Pipeline complete!")
  message("📁 Final output saved to: ", final_output_path)
}
