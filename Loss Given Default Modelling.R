# Load necessary libraries
library(dplyr)
library(caret)
library(ggplot2)
library(glmnet)
library(pROC)

# Load the datasets
# Change the path accordingly if you run this file on the local 
train_data <- read.csv("/Users/yonglaizhu/Desktop/246 Project/df_train.csv")
test_data <- read.csv("/Users/yonglaizhu/Desktop/246 Project/df_test.csv")


# Prepare the datasets
train_data <- subset(train_data, LoanStatus == "CHGOFF")
test_data <- subset(test_data, LoanStatus == "CHGOFF")
train_data <- train_data[ , -(1:15)]
test_data <- test_data[ , -(1:15)]



# ************************ 
#        Method 1 
# ************************ 


# Fit a linear regression model
lgd_model <- lm(GrossChargeOffAmount ~ ., data = train_data)
summary(lgd_model)


# Evaluate the model
predictions <- predict(lgd_model, newdata = test_data)

errors <- predictions - test_data$GrossChargeOffAmount
rmse <- sqrt(mean(errors^2))
mae <- mean(abs(errors))
r_squared <- 1 - sum(errors^2)/sum((test_data$GrossChargeOffAmount - mean(test_data$GrossChargeOffAmount))^2)



# Note: The reason about why we reject method 1 and accept Method 2 will be written on the Google doc Summary.









# ************************ 
#        Method 2
# ************************ 

# Since a significant amount of loans have GrossChargeOffAmount=0, (the problem of high imbalance)
# we implement a classifier for  GrossChargeOffAmount=0 vs GrossChargeOffAmount>0 conditional on default.


# Create a binary outcome variable for classification
train_data$ChargeOffBinary <- ifelse(train_data$GrossChargeOffAmount>0, 1, 0)
test_data$ChargeOffBinary <- ifelse(test_data$GrossChargeOffAmount>0, 1, 0)




# 1st part
# Train Classification Model
classify_model <- glm(ChargeOffBinary~. - GrossChargeOffAmount, data = train_data, family = "binomial")


# Evaluate Classification Model
predictions_binary <- predict(classify_model, test_data, type = "response")
roc_curve <- roc(test_data$ChargeOffBinary, predictions_binary)
print(auc(roc_curve)) 

# Area under curve is calculated around 0.84 (normally 0.8 to 0.9 is considered excellent)

# Find the optimal threshold

# Calculate and Plot Precision-Recall Curve
# The Precision-Recall curve is another tool, especially useful in imbalanced datasets, 
# to assess the performance of different thresholds.
library(precrec)
prec_rec <- evalmod(scores = predictions_binary, labels = test_data$ChargeOffBinary)
autoplot(prec_rec)

# Youden's J Index is a common method to find the optimal threshold from the ROC curve.
optimal_idx <- which.max(roc_curve$sensitivities + roc_curve$specificities - 1)
optimal_threshold <- roc_curve$thresholds[optimal_idx]
print(optimal_threshold)

# Note the optimal threshold is around 0.8

# 2nd part
# Estimate the loss on the default cases where GrossChargeOffAmount>0


# Prepare the datasets
train_data_positive_chargeoff = train_data[train_data$GrossChargeOffAmount>0, ]
test_data_positive_chargeoff = test_data[test_data$GrossChargeOffAmount>0, ]


# Train Regression Model on Positive Charge-off (Charge-off > 0)
lgd_model_positive_chargeooff <- lm(GrossChargeOffAmount~. - ChargeOffBinary, data = train_data_positive_chargeoff)
summary(lgd_model_positive_chargeooff)


# Evaluate Regression Model on Positive Charge-off
predictions_positive_chargeoff <- predict(lgd_model_positive_chargeooff, newdata = test_data_positive_chargeoff)

errors <- predictions_positive_chargeoff - test_data_positive_chargeoff$GrossChargeOffAmount
rmse <- sqrt(mean(errors^2))
mae <- mean(abs(errors))
r_squared <- 1 - sum(errors^2)/sum((test_data_positive_chargeoff$GrossChargeOffAmount - mean(test_data_positive_chargeoff$GrossChargeOffAmount))^2)


# Note: the r-squred value is pretty low, 0.06; should consider variable selection/feature analysis/other methods





# Improve the Regression Model by regularization
# Fit the Lasso Model

features <- as.matrix(train_data_positive_chargeoff[, setdiff(names(train_data_positive_chargeoff), c("GrossChargeOffAmount", "ChargeOffBinary"))])
response <- train_data_positive_chargeoff$GrossChargeOffAmount


# select an optimal lambda value using glmnet
cv_model <- cv.glmnet(features, response, alpha = 1, family = "gaussian") 

optimal_lambda <- cv_model$lambda.min
print(optimal_lambda)


test_features = subset(test_data_positive_chargeoff, select = -GrossChargeOffAmount)
test_features = subset(test_features, select = -ChargeOffBinary)


# Predict with the Lasso Model
predictions_positive_chargeoff <- predict(cv_model, newx = as.matrix(test_features), s = "lambda.min")


# Evaluate the Lasso Model
errors <- predictions_positive_chargeoff - test_data_positive_chargeoff$GrossChargeOffAmount
rmse <- sqrt(mean(errors^2))
mae <- mean(abs(errors))
r_squared <- 1 - sum(errors^2)/sum((test_data_positive_chargeoff$GrossChargeOffAmount - mean(test_data_positive_chargeoff$GrossChargeOffAmount))^2)







# 3rd part
# Apply 


# ****** Things to improve here: threshold selection
# Choose a threshold
predictions_binary <- predict(classify_model, test_data, type = "response")
threshold <- optimal_threshold
predicted_class <- ifelse(predictions_binary > threshold, 1, 0)
test_data$prediction_binary <- predicted_class


# Create a new column predicted_GrossChargeOffAmount
test_data$predicted_GrossChargeOffAmount <- test_data$GrossChargeOffAmount


# Filter for loans predicted to have GrossChargeOffAmount > 0
loans_for_regression <- test_data[test_data$prediction_binary == 1,]


# Apply the regression model to predict amounts for these loans
if(nrow(loans_for_regression) > 0) { # Ensure there are loans to predict for
  predicted_amounts <- predict(lgd_model_positive_chargeooff, newdata = loans_for_regression)
  # Update the predictions in the original dataset
  test_data$predicted_GrossChargeOffAmount[test_data$prediction_binary == 1] <- predicted_amounts
}


# Evaluate the performance
errors <- test_data$predicted_GrossChargeOffAmount - test_data$GrossChargeOffAmount
rmse <- sqrt(mean(errors^2))
mae <- mean(abs(errors))
sse <- sum(errors^2)
sst <- sum((test_data$GrossChargeOffAmount - mean(test_data$GrossChargeOffAmount))^2)
r_squared <- 1 - sse/sst

print(paste("RMSE:", rmse))
print(paste("MAE:", mae))
print(paste("R-squared:", r_squared))



## Summary for modelling on Loss Given Default: Apply the classification model first, which is classify_model, 
## then apply the regression model, which is lgd_model_positive_chargeoff

set.seed(123) # For reproducibility
sampled_test_data <- test_data[sample(nrow(test_data), 1000), ]


