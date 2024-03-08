library("survival")
citation("pROC")
library("ggplot2")
library('ggsurvfit')
library("remotes")


df <- read.csv("/Users/Vincent/Downloads/df_cop (3).csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
df_test <- read.csv("/Users/Vincent/Downloads/df_test_tv.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)


fit <- coxph(Surv(start*12,pmin(end*12,Duration), Event) ~ 1 + ThirdPartyDollars + TermInMonths + GrossApproval + Term.Multiple + Same.State + In.CA + Is.ThirdParty + Missing.Interest 
             + Refinance + Private.Sector + Premier + CORPORATION + INDIVIDUAL + MISSING + PARTNERSHIP  
             + Unemployment.YR + Avg.Home.Price + GDP.Delta.YR + Log.S.P.Open + BorrState.Unemployment + ProjectState.Unemployment
             + BorrState.Income + Missing.Borr.Income + BorrState.GDP + ProjState.GDP + Missing.Borr.GDP
             + Missing.Proj.GDP + BorrState.Vacancy + ProjectState.Vacancy, data=df,cluster=ID)


summary(fit)


prob <- numeric(nrow(df_test))

ALL_TERM <- summary(survfit(fit, newdata = df_test), times = 1:max(df_test$TermInMonths))$surv
ALL_TERM <- matrix(ALL_TERM,nrow = 360, ncol = nrow(df_test))

term_l = df_test$TermInMonths
for (i in 1:nrow(df_test)){
  prob[i] <- ALL_TERM[term_l[i],i]
}
rr <- pROC::roc(ifelse(df_test[,"LoanStatus"] == 1, 1, 0), 1-prob)
rr$auc
plot(rr)

