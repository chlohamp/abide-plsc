

# Utility functions for package management
install_cran_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("Installing CRAN package: %s", pkg))
    install.packages(pkg)
  }
}

install_github_if_missing <- function(pkg, repo) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    if (!requireNamespace("devtools", quietly = TRUE)) {
      install.packages("devtools")
    }
    message(sprintf("Installing GitHub package: %s from %s", pkg, repo))
    devtools::install_github(repo)
  }
}

# -------------------------------
# 1. Load Packages
# -------------------------------
cran_packages <- c(
  "tidyverse", "dplyr", "readr", "here", 
  "ggplot2", "cowplot", "corrplot", "gridExtra", "grid",
  "ExPosition", "TExPosition", "psych", "car", "kableExtra"
)

github_packages <- c("PTCA4CATA", "data4PCCAR")
github_repos <- c("HerveAbdi/PTCA4CATA", "HerveAbdi/data4PCCAR")

# Install & load
invisible(lapply(cran_packages, install_cran_if_missing))
invisible(mapply(install_github_if_missing, github_packages, github_repos))
invisible(lapply(c(cran_packages, github_packages), library, character.only = TRUE))

# -------------------------------
# 2. Directories & Data
# -------------------------------


proj_dir <- here::here()
fig_dir <- file.path(proj_dir, "derivatives/gn-reduced/figures")
out_dir <- file.path(proj_dir, "derivatives/gn-reduced")

covar_df <- read_csv("D:/Hunor/phd_project/Brain_hack/abide-plsc/simulated_covariate.csv")
socio_df <- read_csv("D:/Hunor/phd_project/Brain_hack/abide-plsc/simulated_sociocult.csv")
rsfc_df  <- read_csv("D:/Hunor/phd_project/Brain_hack/abide-plsc/simulated_rsfc.csv")

# -------------------------------
# 3. Residualization Helper
# -------------------------------
run_residualization <- function(y, covar_df) {
  lm_fit <- lm(
    y ~ interview_age +
      as.factor(demo_sex_v2) +
      demo_prnt_age_v2 +
   #   as.factor(demo_prnt_gender_id_v2) +
      demo_prnt_ed_v2_2yr_l +
      demo_prtnr_ed_v2_2yr_l +
      demo_comb_income_v2 +
    #  as.factor(demo_origin_v2) +
      as.factor(site_id_l) +
      as.factor(mri_info_manufacturer) +
      rsfmri_meanmotion,
    data = covar_df,
    na.action = na.omit
  )
  return(as.data.frame(residuals(lm_fit)))
}

# -------------------------------
# 4. Residualization
# -------------------------------
Group1_residuals <- run_residualization(as.matrix(rsfc_df), covar_df)
Group2_residuals <- run_residualization(as.matrix(socio_df), covar_df)

# -------------------------------
# 5. Diagnostic Plots
# -------------------------------
plot_residual_diagnostics <- function(residuals, fit, name, fig_dir) {
  png(file.path(fig_dir, paste0("resi-fit_", name, ".png")), height = 600, width = 1800)
  par(mfrow = c(1, 3))
  
  qqnorm(residuals[,1], main = paste("Q-Q plot for", name, "residuals"))
  qqline(residuals[,1])
  
  hist(residuals[,1], main = paste("Histogram of", name, "residuals"),
       xlab = "Residuals", col = "lightblue")
  
  plot(fit, residuals(fit), main = paste("Fitted vs.", name, "Residuals"))
  
  dev.off()
}

# Example: diagnostics
lm_rsfc <- lm(as.matrix(rsfc_df) ~ ., data = covar_df)  # simplified for demonstration
plot_residual_diagnostics(Group1_residuals, lm_rsfc, "rsfc", fig_dir)

# -------------------------------
# 6. PLSC Analysis
# -------------------------------
data1 <- as.data.frame(Group1_residuals)
data2 <- as.data.frame(Group2_residuals)

# Correlation plot
XY.cor.pearson <- cor(data2, data1)
png(file.path(fig_dir, "correlation_plot.png"), width = 1200, height = 800, res = 300)
corrplot(XY.cor.pearson,
         is.corr = FALSE,
         method = "color",
         tl.cex = 0.2, tl.col = "black",
         cl.pos = "b", cl.cex = 0.3,
         title = "Pearson Correlation",
         cex.main = 0.8,
         mar = c(0,0,1,0),
         lwd = 0.1,
         col = colorRampPalette(c("darkred", "white", "midnightblue"))(6)
)
dev.off()

# Run PLSC
design_matrix <- as.matrix(as.factor(covar_df$site_id_l))
rownames(design_matrix) <- covar_df$src_subject_id

pls_res <- tepPLS(
  data1, data2,
  DESIGN = design_matrix,
  make_design_nominal = TRUE,
  graphs = FALSE
)

summary(pls_res)

# -------------------------------
# 7. Helper Functions for PLSC Outputs
# -------------------------------
get_dim_correlation <- function(pls_res, dim) {
  round(cor(pls_res$TExPosition.Data$lx[,dim],
            pls_res$TExPosition.Data$ly[,dim]), 3)
}

plot_score <- function(pls_res, dim, correlation, fig_dir) {
  latvar <- cbind(pls_res$TExPosition.Data$lx[,dim],
                  pls_res$TExPosition.Data$ly[,dim])
  colnames(latvar) <- c(paste("Latent X dim", dim),
                        paste("Latent Y dim", dim))
  
  lv_map <- createFactorMap(latvar, title = paste("Correlation =", correlation))
  score_plot <- lv_map$zeMap_background + lv_map$zeMap_dots
  
  ggsave(score_plot, file.path(fig_dir, paste0("Score_Dim", dim, ".png")),
         width = 6, height = 6, dpi = 300)
}

# Example usage
cor_dim3 <- get_dim_correlation(pls_res, 3)
plot_score(pls_res, 3, cor_dim3, fig_dir)
