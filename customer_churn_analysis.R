# Calling necessary packages
library(tidyverse)
library(DataExplorer)
library(caret)
library(gridExtra)
library(MASS)
library(scales)
library(MLeval)

# Read input file
telcom <-
  read.csv("D:/Path/WA_Fn-UseC_-Telco-Customer-Churn.csv",
           stringsAsFactors = FALSE)

# Basic stats
summary(telcom)
str(telcom)
introduce(telcom)
plot_intro(telcom)

# Handle missing data
telcom <- telcom[complete.cases(telcom),]
telcom$SeniorCitizen <- as.factor(telcom$SeniorCitizen)

summary(telcom)

# Exploratory data analysis
plot_bar(telcom)
plot_histogram(telcom)

# What are the categorical fields in the data?
cats <-
  names(telcom %>% dplyr::select(-customerID) %>% select_if(is.factor))

# Plotting all variables with Churn
cat_plot <- function(col_name) {
    ggplot(telcom,aes(x = get(col_name), fill = Churn)) +
    geom_bar(aes(y = (..count..)/sum(..count..))) + 
    scale_y_continuous(labels = scales::percent) +
    coord_flip() +
    scale_fill_manual(values = c("#a5d685", "#f5897f")) +
    theme(legend.position = "bottom") +
    xlab(col_name) +
    ylab("Number of customers")
  }

attach(telcom)

cat_plot("Churn")

grid.arrange(
  cat_plot("gender"),
  cat_plot("SeniorCitizen"),
  cat_plot("Partner"),
  cat_plot("Dependents"),
  nrow = 2
)

grid.arrange(
  cat_plot("PhoneService"),
  cat_plot("MultipleLines"),
  cat_plot("InternetService"),
  cat_plot("OnlineSecurity"),
  cat_plot("OnlineBackup"),
  cat_plot("DeviceProtection"),
  cat_plot("TechSupport"),
  cat_plot("StreamingTV"),
  cat_plot("StreamingMovies"),
  nrow = 3, ncol = 3
)

hist_plot <- function(col_name){
  plot <- telcom %>% 
    ggplot(aes(x=get(col_name),fill=Churn)) +
    geom_histogram(alpha=0.8,bins = 50) +
    scale_fill_manual(values = c("#a5d685", "#f5897f")) +
    xlab(col_name)  +
    ylab("Number of customers")
  return(plot)
}

grid.arrange(
  hist_plot("tenure"),
  hist_plot("MonthlyCharges"),
  hist_plot("TotalCharges"),
  nrow=2
)


# Feature Engineering
internet_cols <- c("OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies")

telcom <- telcom %>% 
  mutate(MultipleLines = str_replace(MultipleLines,"No phone service", "No")) %>%
  mutate_at(vars(internet_cols), ~ str_replace(., "No internet service", "No")) %>% 
  mutate(tenure_group = case_when(tenure <= 12 ~ "1 year", # grouping tenure into smaller groups
                                  tenure > 12 & tenure <= 24 ~ "1-2 years",
                                  tenure > 24 & tenure <= 36 ~ "2-3 years",
                                  tenure > 36 & tenure <= 48 ~ "3-4 years",
                                  tenure > 48 & tenure <= 60 ~ "4-5 years",
                                  tenure > 60 ~ "5+ years",
                                  TRUE ~ NA_character_)) %>% 
  dplyr::select(-customerID,-tenure,-TotalCharges)
                                  
# Exploratory analysis
grid.arrange(
  cat_plot("Contract"),
  cat_plot("PaperlessBilling"),
  cat_plot("PaymentMethod"),
  cat_plot("tenure_group"),
  nrow = 2
)

# Pre-processing for the model

set.seed(3456)

# Split into train and test
trainIndex <- createDataPartition(telcom$Churn, p = .7, 
                                  list = FALSE, 
                                  times = 1)
train <- telcom[ trainIndex,]
test  <- telcom[-trainIndex,]

# Cross validation method
ctrl <- trainControl(method="cv", 
                     summaryFunction=twoClassSummary, 
                     classProbs=TRUE,
                     savePredictions = TRUE)

# Model fitting
# Logistic Regression
glm_fit <- train(Churn ~ ., data = train,
                method = "glm",
                family = "binomial",
                preProc=c("center", "scale"), 
                trControl=ctrl)

summary(glm_fit)

# Predictions

predictions <- predict(glm_fit, test)
confusionMatrix(predictions, as.factor(test$Churn), mode = "prec_recall")

# Variable importance
plot(varImp(glm_fit))

res <- evalm(glm_fit)
res$roc

# Decision Tree
rpart_fit <- train(Churn ~ ., data = train,
                 method = "rpart")

summary(rpart_fit)

# Predictions

predictions <- predict(rpart_fit, test)
confusionMatrix(predictions, as.factor(test$Churn), mode = "prec_recall")

# Variable Importance
plot(varImp(rpart_fit))
