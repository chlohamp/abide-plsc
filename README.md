# ABIDE PLSC Analysis

This repo contains the scripts used to run a partial least squares correlation (PLSC) analysis in the Autism Brain Imaging Data Exchange (ABIDE). This project was concieved at the 2025 Flux BrainHack. For more information regarding the team who worked on the project, please see [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Scripts
- `plsc.Rmd`: main PLSC analysis script
- `beta.ipynb`: extracts the correlation coefficients using the Yeo atlas
- `plsc-ppt.ipynb`: determines the number of participants that have phenotypic data

## Simulated Data
To test the PLSC analysis script, we geneterated simulated csv files (`simulated_covariate.csv`, `simulated_rsfc.csv`, `simulated_sociocult.csv`) using the ABCD dataset. 
