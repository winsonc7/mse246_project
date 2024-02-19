library(caret)
library(dplyr)
library(GGally)
library(ggplot2)
library(pROC) 


set.seed(123)

df <- read.csv("/Users/Vincent/Downloads/MS&E 246 Project/MS&E 246 Data Updated 3/df_train.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
df_test <- read.csv("/Users/Vincent/Downloads/MS&E 246 Project/MS&E 246 Data Updated 3/df_test.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
levels(df$LoanStatus) <- c(1,0)
levels(df_test$LoanStatus) <- c(1,0)

# base model for reference and variable selection
logisticModel <- train(LoanStatus ~ 1 + ThirdPartyDollars + TermInMonths + GrossApproval + Term.Multiple + Same.State + In.CA + Is.ThirdParty + Missing.Interest 
                       + Refinance + Delta + Private.Sector + Premier + CORPORATION + INDIVIDUAL + MISSING + PARTNERSHIP + SP500.YR 
                       + Unemployment.YR + Avg.Home.Price + GDP.Delta.YR + Log.S.P.Open + BorrState.Unemployment + ProjectState.Unemployment
                       + BorrState.Income + Missing.Borr.Income + Missing.Proj.Income + BorrState.GDP + ProjState.GDP + Missing.Borr.GDP
                       + Missing.Proj.GDP + BorrState.Vacancy + ProjectState.Vacancy
                         , data = df, method = "glm",
                       family = "binomial")
summary(logisticModel)

logistic_predictions <- predict(logisticModel, newdata = df_test, type = "prob")

roc_curve_noInt <- roc(response = df_test$LoanStatus, predictor = logistic_predictions[,1])


# accuracy <- mean(logistic_predictions == df_test$LoanStatus)
# print(accuracy)

# base model with interaction terms added 
logisticModel_int <- train(LoanStatus ~ 1 + ThirdPartyDollars + TermInMonths + GrossApproval + Term.Multiple + Same.State + In.CA + Is.ThirdParty + Missing.Interest 
                       + Refinance + Delta + Private.Sector + Premier + CORPORATION + INDIVIDUAL + MISSING + PARTNERSHIP + SP500.YR 
                       + Unemployment.YR + Avg.Home.Price + GDP.Delta.YR + Log.S.P.Open + BorrState.Unemployment + ProjectState.Unemployment
                       + BorrState.Income + Missing.Borr.Income + Missing.Proj.Income + BorrState.GDP + ProjState.GDP + Missing.Borr.GDP
                       + Missing.Proj.GDP + BorrState.Vacancy + ProjectState.Vacancy
                       
                       + Same.State:ThirdPartyDollars + Same.State:TermInMonths + Same.State:GrossApproval + Same.State:Term.Multiple + Same.State:Unemployment.YR + Same.State:Avg.Home.Price + Same.State:GDP.Delta.YR 
                       + Same.State:Log.S.P.Open + Same.State:BorrState.Unemployment + Same.State:ProjectState.Unemployment + Same.State:BorrState.Income + Same.State:Missing.Borr.Income + Same.State:Missing.Proj.Income 
                       + Same.State:BorrState.GDP + Same.State:ProjState.GDP + Same.State:Missing.Borr.GDP + Same.State:Missing.Proj.GDP + Same.State:BorrState.Vacancy + Same.State:ProjectState.Vacancy
                         
                         
                       + Missing.Interest:ThirdPartyDollars + Missing.Interest:TermInMonths + Missing.Interest:GrossApproval + Missing.Interest:Term.Multiple + Missing.Interest:Unemployment.YR + Missing.Interest:Avg.Home.Price + Missing.Interest:GDP.Delta.YR + Missing.Interest:Log.S.P.Open 
                       + Missing.Interest:BorrState.Unemployment + Missing.Interest:ProjectState.Unemployment + Missing.Interest:BorrState.Income + Missing.Interest:Missing.Borr.Income + Missing.Interest:Missing.Proj.Income + Missing.Interest:BorrState.GDP + Missing.Interest:ProjState.GDP 
                       + Missing.Interest:Missing.Borr.GDP + Missing.Interest:Missing.Proj.GDP + Missing.Interest:BorrState.Vacancy + Missing.Interest:ProjectState.Vacancy
                       
                       + INDIVIDUAL:ThirdPartyDollars + INDIVIDUAL:TermInMonths + INDIVIDUAL:GrossApproval + INDIVIDUAL:Term.Multiple + INDIVIDUAL:Unemployment.YR + INDIVIDUAL:Avg.Home.Price + INDIVIDUAL:GDP.Delta.YR + INDIVIDUAL:Log.S.P.Open + INDIVIDUAL:BorrState.Unemployment 
                       + INDIVIDUAL:ProjectState.Unemployment + INDIVIDUAL:BorrState.Income + INDIVIDUAL:Missing.Borr.Income + INDIVIDUAL:Missing.Proj.Income +INDIVIDUAL:BorrState.GDP + INDIVIDUAL:ProjState.GDP + INDIVIDUAL:Missing.Borr.GDP
                       + INDIVIDUAL:Missing.Proj.GDP + INDIVIDUAL:BorrState.Vacancy + INDIVIDUAL:ProjectState.Vacancy
                       
                       , data = df, method = "glm",
                       family = "binomial")

logistic_predictions_int <- predict(logisticModel_int, newdata = df_test, type = 'prob')

roc_curve_int <- roc(response = df_test$LoanStatus, predictor = logistic_predictions_int[,1])
plot(roc_curve_noInt, main = "ROC Curves", col = "blue", xlim = c(1, 0), ylim = c(0, 1))
lines(roc_curve_int,col="red", xlim = c(1, 0), ylim = c(0, 1))
legend(0.5, 0.3, legend=c("Baseline", "With Interaction"),
       col=c("blue", "red"), lty=1:2, cex=0.8)

text(0.5, 0.6, paste("AUC =", round(auc(roc_curve_noInt), 4)), col = "blue", cex = 1.2)
text(0.5, 0.4, paste("AUC =", round(auc(roc_curve_int), 4)), col = "red", cex = 1.2)



