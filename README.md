# dnaPipelineR
**dnaPipelineR** is a prototype semi-supervised text annotation pipeline designed to reduce the manual burden of Discourse Network Analysis (DNA). It automatically extracts and labels statements from political transcripts, making it easier for researchers to build discourse networks without line-by-line manual coding.

The pipeline was developed for application to UK Parliamentary Hansard debates on labour-law remedies, but its modular design makes it adaptable to other domains. It combines rule-based heuristics, pretrained language models, and machine learning classifiers to annotate transcripts with multiple attributes.

Specifically, dnaPipelineR:
- Identifies actors (who is speaking, including organisational affiliation).
- Segments debates into statements (claims or opinions).
- Classifies each statement into one of 21 concepts and 6 labour rights, following a specialist codebook developed by Prof. Aristea Koukiadaki (may be added to the repository subject to approval).
- Assigns a sentiment label (positive/neutral/negative) as a proxy for stance.

The output is a structured `.csv file` where each row is a statement annotated with:
- Actor and organisation
- Statement text
- Concept label
- Right label
- Sentiment label
- Confidence scores

The pipeline is designed with future integration into the Java-based DNA software Discourse [Discourse Network Analyzer](https://github.com/leifeld/dna) (DNA, v3.0.11; Leifeld, 2024) in mind, enabling researchers to map actor–concept networks more quickly and consistently than through manual annotation alone. For additional materials including: the manually annotated dataset used for training models, code and results from experiments on classifiers, and an example output folder generated from the pipeline please visit the separate repository [dnaPipelineR-materials](https://github.com/sarahtunmore/dnaPipelineR-materials).

**dnaPipelineR** was created as part of a Master’s Extended Research Project (MSc Data Science, University of Manchester, 2025) under the supervision of Prof. Philip Leifeld.

# The Pipeline
The pipeline consists of is three main scripts (found in `R/` folder):

- `check_pipeline`: This script will check that all requirements are met for the pipeline to run. If any requirements are not met, it will flag this in the output.
*For more experienced programmers*, you may just want to run this check and go install/update the necessary packages yourself in order to run the pipeline.
*For less experienced programmers/new users to the pipeline*, you can still use this to see if all requirements are met and if they are not you will be asked to `run setup_pipeline.R` which will run the installs for you.
*For returning users to the pipeline*, this script can serve as a quick sanity check before running the pipeline to avoid configuration errors.

- `setup_pipeline`: This script is designed for *new* users to the pipeline, primarily to create a conda environment **dna-pipeline** and install the necessary packages into it. It checks that conda is present, and if not will download Minicoda. It also checks that necessary R packages are installed and loaded. Once run this script will not need to be run again, any debugging can be performed with check_pipeline.

- `run_pipeline`: This script runs the pipeline fully. It uses pretrained models (found in `inst/models`) and prewritten .py functions (found in `inst/python`). Once run, it will ask the user to select a text file they want to analyse, then the pipeline will run and output a spreadsheet with the annotations, which will be saved the users working directory.

# Repository Structure
```plaintext
dnaPipelineR/
├── R/
│   ├── check_pipeline.R       # Verifies requirements
│   ├── setup_pipeline.R       # Creates conda env, installs dependencies
│   └── run_pipeline.R         # Runs the full pipeline
├── inst/
│   ├── models/                # Pretrained classifiers (MLP, SVM, encoders)
│   ├── python/                # Python helper functions (via reticulate)
├── DESCRIPTION                # R package metadata
├── NAMESPACE                  # Exported functions
└── dnaPipelineR.Rproj         # RStudio project file
```


# Requirements
- R (≥ 4.2.0)
- RStudio
- Conda / Miniconda (setup script installs if missing)
- Python 3.9+ (managed automatically by setup script)

# How to install and Use
1) Install the package
```r
# install if you haven't already:
install.packages("remotes")  # if needed
remotes::install_github("sarahtunmore/dnaPipelineR")

library(dnaPipelineR)
```

2) (Optional but reccommended) Quick Environment Check
```r
# returns TRUE if everything looks good; otherwise tells you what’s missing
check_pipeline_dependencies()       # checks R pkgs, rDNA, Conda, and required Python pkgs
```

3) One-time setup (creates Conda env + installs Python dependencies)
```r
# run this if step 2 shows anything missing OR if you're a first time user
setup_pipeline()                    # creates env "dna-pipeline", installs Python packages, spaCy model, rDNA, etc.
```

4) Run the pipeline: prompts you to choose a `.txt` file via file dialogue
```r
# loads models from inst/models and Python helpers from inst/python
# prompts you to pick a .txt file (Hansard transcript)
run_pipeline()
```
When finished, a CSV named something like `pipeline_output_<yourfilename>.csv` is saved to your working directory


# Input and Ouput
## Input
A plain text Hansard transcript (`.txt`)

## Output
Each module outputs its own `CSV` file for transparency
- `actor_organisation_ouput.csv` - Output from actor/organisation assignment module
- `actor_org_statements_confidence.csv` – Output from statement segmentation module.
- `statement_concepts.csv` – Predictions from concept classification module.
- `statement_rights.csv` – Predictions from right classification module.
- `actor_org_statements_with_statements.csv` - Output from sentiment Analysis module.

The final stage of the pipeline outputs a single CSV file:
- `pipeline_output_<yourfilename>.csv` – Consolidated output for use in Discourse Network Analyzer, uses the text name of TXT debate inputted.

Columns include:
- `speaker_org`: Speaker + organisational affiliation (parsed from Hansard headings).
- `statement`: The extracted claim/utterance.
- `confidence`: confidence score generated from statement classifier, how confident model is that the text is a statement.
- `concept`: 21 class concept label(s) from codebook
- `right`: 6 class right label from codebook
- `sentiment_label`: positive/neutral/negative sentiment

For an example of outputs generated from the pipeline on a debate, please see the additional materials repo: [dnaPipelineR-materials/outputs](https://github.com/sarahtunmore/dnaPipelineR-materials/tree/main/outputs)

# What the Pipeline does
1. **Actor / organisation extraction**
Rule-based parsing of Hansard speaker headings into `speaker_org`, plus text block separation.

2. **Statement segmentation**
MLP classifier over SBERT embeddings identifies statements vs. non-statements, with light discourse heuristics.

3. **Concept classification**
Linear SVM on SBERT embeddings assigns a 21-class concept (returns Top-1 and Top-3 candidates).

4. **Right classification**
Linear SVM (SBERT embeddings ± POS features) assigns a single right label (6 classes).

5. **Sentiment analysis**
Transformer-based sentiment model produces positive / neutral / negative labels.

All models are bundled in `inst/models/`, and Python helpers live in `inst/python/` (called through `reticulate`).

# Troubleshooting
*“Conda not found”*:
Run `setup_pipeline()` (it will install Miniconda if missing). If you have multiple Condas, ensure the one you want is first on PATH, or let `setup_pipeline()` handle it.

*macOS + rJava issues*: 
Install Java (Adoptium Temurin 17 or Oracle JDK), then in R: `install.packages("rJava", type="source")`. Reopen RStudio.

*Windows antivirus blocks Miniconda*: 
Temporarily allow the installer; then rerun `setup_pipeline()`.

*Permission / write errors*: 
Check your working directory (`getwd()`); choose a folder you can write to.

*Want to recheck everything*: 
`check_pipeline_dependencies()` at any time. If something’s off, run `setup_pipeline()` again- it will not install/update something again if it already exists in the environment.

# Notes
- optimised for UK Hansard labour-law debates. Transfer to other corpora may require tweaks to speaker-heading heuristic and/or retraining
- The concept classifier provides its Top-3 best estimate when its confidence is low (below a threshold), this supports a semi-automatic workflow where a human can pick the best plausible option
- The prototype is not yet integrated with the Discourse Network Analyzer software, but its implementation in Rstudio supports further project development via the `rDNA` package (which allows Discourse Network Analyzer and Rstudio talk to eachother)

# Credits
**Codebook**: Prof Aristea Koukiadaki

**Supervision**: Philip Leifeld

**Core Libraries**: SBERT, HuggingFace Transformers, Scikit-learn, spaCy, reticulate

**Transcripts**: [Hansard](https://hansard.parliament.uk/)
