library("survival")
citation("pROC")
library("ggplot2")
library('ggsurvfit')
library("remotes")


df <- read.csv("/Users/Vincent/Downloads/df_cop (2).csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
df_test <- read.csv("/Users/Vincent/Downloads/df_cop_test (2).csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)

fit <- coxph(Surv(start, pmin(end,Duration/12.0), Event) ~ 1 + ThirdPartyDollars + TermInMonths + GrossApproval + Term.Multiple + Same.State + In.CA + Is.ThirdParty + Missing.Interest 
             + Refinance + Private.Sector + Premier + CORPORATION + INDIVIDUAL + MISSING + PARTNERSHIP  
             + Unemployment.YR + Avg.Home.Price + GDP.Delta.YR + Log.S.P.Open + BorrState.Unemployment + ProjectState.Unemployment
             + BorrState.Income + Missing.Borr.Income + BorrState.GDP + ProjState.GDP + Missing.Borr.GDP
             + Missing.Proj.GDP + BorrState.Vacancy + ProjectState.Vacancy, data=df, method = "breslow")


summary(fit)

survival_prob <- survfit(fit)

# Plot survival curve
plot(survival_prob, main = "Survival Curve", xlab = "Time (Years)", ylab = "Survival Probability", col = "blue", xlim = c(0, 20), ylim = c(0.65, 1))


###### IGNORE THE FOLLOWING CODE (COULD BE INCORRECT) #######


df_test <- read.csv("/Users/Vincent/Downloads/df_cop_test.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
df_test$end = pmin(df_test$end,df_test$Duration/12.0)

predicted_values <- predict(fit, newdata = df_test)
predicted_values <- predict(fit, newdata = df_test, type = "lp")
ex <- exp(predicted_values)
prob <- 1-exp(-exp(predicted_values))
rr <- pROC::roc(ifelse(df_test[,"Event"] == 1, 1, 0), prob)
rr$auc
plot(rr)