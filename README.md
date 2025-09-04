# ABIDE PLSC Analysis

This repo contains the scripts used to run a partial least squares correlation (PLSC) analysis in the Autism Brain Imaging Data Exchange (ABIDE). This project was concieved at the 2025 Flux BrainHack. For more information regarding the team who worked on the project, please see [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Scripts
- `plsc.Rmd`: Main PLSC analysis script
- `beta.py`: Original script that denoises fMRIPrep data and creates a correlation matrix
- `plsc-ppt.ipynb`: Determines the number of participants that have phenotypic data
- `beta_retrieval_from_bold.py`: Extracts the correlation coefficients using the Yeo atlas -> Outputs a csv file with row = participants and columns = the correlation between different network

## Simulated Data
To test the PLSC analysis script, we geneterated simulated csv files (`simulated_covariate.csv`, `simulated_rsfc.csv`, `simulated_sociocult.csv`) using the ABCD dataset. 
