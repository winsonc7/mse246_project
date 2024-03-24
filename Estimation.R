## Copy from Github files
library(caret)
library(dplyr)
library(GGally)
library(ggplot2)
library(pROC) 


set.seed(123)

df <- read.csv("/Users/yonglaizhu/Desktop/246 Project/df_train.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
df_test <- read.csv("/Users/yonglaizhu/Desktop/246 Project/df_test.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)

levels(df$LoanStatus) <- c(1,0)
levels(df_test$LoanStatus) <- c(1,0)

# base model for reference and variable selection
logisticModel <- train(LoanStatus ~ 1 + ThirdPartyDollars + TermInMonths + GrossApproval + Term.Multiple + Same.State + In.CA + Is.ThirdParty + Missing.Interest 
                       + Refinance + Delta + Private.Sector + Premier + CORPORATION + INDIVIDUAL + MISSING + PARTNERSHIP  
                       + Unemployment.YR + Avg.Home.Price + GDP.Delta.YR + Log.S.P.Open + BorrState.Unemployment + ProjectState.Unemployment
                       + BorrState.Income + Missing.Borr.Income + Missing.Proj.Income + BorrState.GDP + ProjState.GDP + Missing.Borr.GDP
                       + Missing.Proj.GDP + BorrState.Vacancy + ProjectState.Vacancy
                       , data = df, method = "glm",
                       family = "binomial")
summary(logisticModel)

logistic_predictions <- predict(logisticModel, newdata = df_test, type = "prob")



############################################################################
n_samples <- 100 # Number of times to repeat the sampling, e.g., 100
total_losses <- numeric(n_samples)

for(j in 1:n_samples) {
  ## Estimate total loss distribution by randomly sampling portfolios consisting of 1,000 loans from the test set
  sampled_test_data <- df_test[sample(nrow(df_test), 1000), ]
  
  # Whether it is default
  default_probs <- predict(logisticModel, newdata = sampled_test_data, type = "prob")
  default_preds <- ifelse(default_probs$CHGOFF > 0.4, 1, 0) # Adjust threshold as necessary
  
  
  # Initialize a vector to store estimated losses
  sampled_test_data$EstimatedLoss <- numeric(nrow(sampled_test_data))
  sampled_test_data$ChargeOffBinary <- numeric(nrow(sampled_test_data))
  
  # For loans predicted to default, predict GrossChargeOffAmount category
  for(i in which(default_preds == 1)) {
    chargeoff_prob <- predict(classify_model, sampled_test_data[i, ], type = "response")
    chargeoff_pred <- ifelse(chargeoff_prob > optimal_threshold, 1, 0)
    sampled_test_data$ChargeOffBinary[i] <- chargeoff_pred
    
    if(chargeoff_pred == 1) { # If GrossChargeOffAmount is predicted to be > 0
      estimated_loss <- predict(lgd_model_positive_chargeooff, sampled_test_data[i, ])
      sampled_test_data$EstimatedLoss[i] <- estimated_loss
    }
  }
  
  total_losses[j] <- sum(sampled_test_data$EstimatedLoss) # Hypothetical
  print(paste("Total Estimated Loss:", total_losses[j]))
  
}

############################################################################

hist(total_losses, breaks = 50, main = "Distribution of Estimated Loss")
summary(total_losses)



## VaR and average VaR (CVaR)

# Calculate VaR at the 95% and 99% levels
VaR_95 <- quantile(total_losses, 0.95)
VaR_99 <- quantile(total_losses, 0.99)

# Calculate AVaR (Expected Shortfall) at the 95% and 99% levels
AVaR_95 <- mean(total_losses[total_losses > VaR_95])
AVaR_99 <- mean(total_losses[total_losses > VaR_99])

print(paste("VaR 95%:", VaR_95))
print(paste("VaR 99%:", VaR_99))
print(paste("AVaR 95%:", AVaR_95))
print(paste("AVaR 99%:", AVaR_99))


# Estimate the confidence intervals
# Bootstrapping for 95% VaR
bootstrap_samples <- 1000 # Number of bootstrap samples
VaR_95_samples <- numeric(bootstrap_samples)

for (i in 1:bootstrap_samples) {
  # Sample with replacement from total_losses
  sample_losses <- sample(total_losses, replace = TRUE)
  VaR_95_samples[i] <- quantile(sample_losses, 0.95)
}

# Calculate confidence intervals for VaR 95%
VaR_95_CI <- quantile(VaR_95_samples, probs = c(0.025, 0.975))
print(paste("95% Confidence Interval for VaR 95%:", VaR_95_CI))

####################################################
n_bootstraps <- 1000
VaR_95_samples <- numeric(n_bootstraps)
CVaR_95_samples <- numeric(n_bootstraps)


for (i in 1:n_bootstraps) {
  bootstrap_sample <- sample(total_losses, size = length(total_losses), replace = TRUE)
  
  # Recalculate VaR and CVaR for the bootstrap sample
  bootstrap_VaR_95 <- quantile(bootstrap_sample, 0.95)
  bootstrap_CVaR_95 <- mean(bootstrap_sample[bootstrap_sample >= bootstrap_VaR_95])
  
  VaR_95_samples[i] <- bootstrap_VaR_95
  CVaR_95_samples[i] <- bootstrap_CVaR_95
}

# Calculate the 95% confidence intervals for VaR 95% and CVaR 95%
VaR_95_CI <- quantile(VaR_95_samples, probs = c(0.025, 0.975))
CVaR_95_CI <- quantile(CVaR_95_samples, probs = c(0.025, 0.975))

print(paste("95% Confidence Interval for VaR 95%: [", VaR_95_CI[1], ", ", VaR_95_CI[2], "]"))
print(paste("95% Confidence Interval for CVaR 95%: [", CVaR_95_CI[1], ", ", CVaR_95_CI[2], "]"))

####################################################

VaR_99_samples <- numeric(n_bootstraps)
CVaR_99_samples <- numeric(n_bootstraps)

for (i in 1:n_bootstraps) {
  bootstrap_sample <- sample(total_losses, size = length(total_losses), replace = TRUE)
  
  # Recalculate VaR and CVaR for the bootstrap sample
  bootstrap_VaR_99 <- quantile(bootstrap_sample, 0.99)
  bootstrap_CVaR_99 <- mean(bootstrap_sample[bootstrap_sample >= bootstrap_VaR_99])
  
  VaR_99_samples[i] <- bootstrap_VaR_99
  CVaR_99_samples[i] <- bootstrap_CVaR_99
}

# Calculate the 95% confidence intervals for VaR 99% and CVaR 99%
VaR_99_CI <- quantile(VaR_99_samples, probs = c(0.025, 0.975))
CVaR_99_CI <- quantile(CVaR_99_samples, probs = c(0.025, 0.975))

print(paste("95% Confidence Interval for VaR 99%: [", VaR_99_CI[1], ", ", VaR_99_CI[2], "]"))
print(paste("95% Confidence Interval for CVaR 99%: [", CVaR_99_CI[1], ", ", CVaR_99_CI[2], "]"))

####################################################
