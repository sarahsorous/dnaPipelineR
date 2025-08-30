# dna-automatic-annotation-pipeline
A prototype for a semi-supervised automatic annotation pipeline that assists annotators in extracting and annotating pieces of text from transcripts, with a primary application to UK Parliamentary Hansard debates and discussions of labour law remedies for purposes of discourse network analysis (dna).

When a user runs the pipeline it will: take .txt files a user selects and automatically extract who said the text (actors), claims and opinions they express (statements), how they frame it (concepts), what area they are discussing (right), and what mood they express (sentiment). The output will be a spreadsheet with the extracted statements and annotation.

Designed in Rstudio for future integration with the Discourse Network Analyzer (DNA) software (version 3.0.11; Leifeld, 2024). Created as part of my Master's Extended Research Project (Msc Data Science) at the University of Manchester under the supervsion of Phillip Leifeld.

# The Pipeline
The code is broken into 3 R files:

**1. check_pipeline.R**

**2. setup_pipeline.R**

**3. run_pipeline.R**



**check_pipeline**: This script will check that all requirements are met for the pipeline to run. If any requirements are not met, it will flag this in the output.

*For more experienced programmers*, you may just want to run this check and go install/update the necessary packages yourself in order to run the pipeline.

*For less experienced programmers/new users to the pipeline*, you can still use this to see if all requirements are met and if they are not you will be asked to run setup_pipeline.R which will run the installs for you.

*For returning users to the pipeline*, this script can serve as a quick sanity check before running the pipeline to avoid configuration errors.

**setup_pipeline**: This script is deigned for *new* users to the pipeline, primarily to create a conda environment 'dna-pipeline' and install the necessary packages into it. It checks that conda is present, and if not will download minicoda. It also checks that necessary R packages are installed and loaded. Once run this script will not need to be run again, any debgugging can be performed with check_pipeline.

**run_pipeline**: This script runs the pipeline fully. It uses pretrained models (found in inst > models) and prewritten .py functions (found in inst > python). Once run, it will ask the user to select a text file they want to analyse, then the pipeline will run and output a spreadsheet with the annotations, which will be saved the users working directory.
