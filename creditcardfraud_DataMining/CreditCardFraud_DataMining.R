library(ggplot2)
library(dplyr)
library(tidyr)
library(corrplot)
library(ROSE)
library(tree)
library(Metrics)
library(gplots)
library(caret)
library(caretEnsemble)
library(randomForest)
library(xgboost)

set.seed(123)

trans_data <- read.csv("card_transdata.csv")

summary(trans_data)

correlation_matrix <- cor(trans_data)
corrplot(correlation_matrix, method = "color")

par(mfrow = c(1, 3), mar = c(4, 4, 2, 2))
plot(trans_data$distance_from_home, main = "distance_from_home", xlab = "", ylab = "")
plot(trans_data$distance_from_last_transaction, main = "distance_from_last_transaction", xlab = "", ylab = "")
plot(trans_data$ratio_to_median_purchase_price, main = "ratio_to_median_purchase_price", xlab = "", ylab = "")

columns_to_plot <- c('repeat_retailer', 'used_chip', 'used_pin_number', 'online_order', 'fraud')
par(mfrow = c(1, length(columns_to_plot)), mar = c(4, 4, 2, 2))
for (column in columns_to_plot) {
  counts <- table(trans_data[[column]])
  barplot(counts, names.arg = c('0 (No)', '1 (Yes)'), main = column, xlab = "", ylab = "Amount")
}

par(mfrow = c(1, 3), mar = c(4, 4, 2, 2))
hist(trans_data$distance_from_home, main = "distance_from_home", xlab = "distance_from_home", col = "blue", alpha = 0.7)
hist(trans_data$distance_from_last_transaction, main = "distance_from_last_transaction", xlab = "distance_from_last_transaction", col = "green", alpha = 0.7)
hist(trans_data$ratio_to_median_purchase_price, main = "ratio_to_median_purchase_price", xlab = "ratio_to_median_purchase_price", col = "orange", alpha = 0.7)

value_iqr <- function(df, variable) {
  q1 <- quantile(df[[variable]], probs = 0.25)
  q3 <- quantile(df[[variable]], probs = 0.75)
  IQR <- q3 - q1
  upper_limit <- q3 + 3 * IQR
  return(upper_limit)
}

upper_limit_distance_from_home <- value_iqr(trans_data, "distance_from_home")
upper_limit_distance_from_last_transaction <- value_iqr(trans_data, "distance_from_last_transaction")
upper_limit_ratio_to_median_purchase_price <- value_iqr(trans_data, "ratio_to_median_purchase_price")

par(mfrow = c(1, 3), mar = c(4, 4, 2, 2))
hist(trans_data$distance_from_home, main = "distance_from_home", xlab = "distance_from_home", col = "blue", alpha = 0.7)
abline(v = mean(trans_data$distance_from_home), col = "red", lty = 2)
abline(v = upper_limit_distance_from_home, col = "green", lty = 3)
hist(trans_data$distance_from_last_transaction, main = "distance_from_last_transaction", xlab = "distance_from_last_transaction", col = "green", alpha = 0.7)
abline(v = mean(trans_data$distance_from_last_transaction), col = "red", lty = 2)
abline(v = upper_limit_distance_from_last_transaction, col = "green", lty = 3)
hist(trans_data$ratio_to_median_purchase_price, main = "ratio_to_median_purchase_price", xlab = "ratio_to_median_purchase_price", col = "orange", alpha = 0.7)
abline(v = mean(trans_data$ratio_to_median_purchase_price), col = "red", lty = 2)
abline(v = upper_limit_ratio_to_median_purchase_price, col = "green", lty = 3)

trans_data_clean <- trans_data[trans_data$distance_from_home < upper_limit_distance_from_home &
                                 trans_data$distance_from_last_transaction < upper_limit_distance_from_last_transaction &
                                 trans_data$ratio_to_median_purchase_price < upper_limit_ratio_to_median_purchase_price, ]

dim(trans_data_clean)

columns_scaled <- c('distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price')
df_processing_scaler <- as.data.frame(scale(trans_data_clean[columns_scaled]))
df_processing_scaler <- cbind(df_processing_scaler, trans_data_clean[setdiff(names(trans_data_clean), columns_scaled)])

head(df_processing_scaler)

cat("Number of registers:\n", table(trans_data_clean$fraud), "\n")

X <- trans_data_clean[, !names(trans_data_clean) %in% c("fraud")]
y <- trans_data_clean$fraud

undersampler <- ovun.sample(fraud ~ ., data = trans_data_clean, method = "under", N = nrow(trans_data_clean)$fraud)
trans_data_usmp <- data.frame(undersample(trans_data_clean, undersampler))

cat("Number of registers:\n", table(trans_data_usmp$fraud), "\n")

set.seed(42)
split_index <- createDataPartition(trans_data_usmp$fraud, p = 0.8, list = FALSE)
X_train <- trans_data_usmp[split_index, !names(trans_data_usmp) %in% c("fraud")]
y_train <- trans_data_usmp$fraud[split_index]
X_test_validation <- trans_data_usmp[-split_index, !names(trans_data_usmp) %in% c("fraud")]
y_test_validation <- trans_data_usmp$fraud[-split_index]

split_index_test <- createDataPartition(y_test_validation, p = 0.5, list = FALSE)
X_test <- X_test_validation[split_index_test, ]
y_test <- y_test_validation[split_index_test]
X_validation <- X_test_validation[-split_index_test, ]
y_validation <- y_test_validation[-split_index_test]

dim(X_test)

set.seed(0)
random_forest <- randomForest(fraud ~ ., data = data.frame(cbind(y_train, X_train)), ntree = 10, mtry = ncol(X_train), nodesize = 2)

y_train_pred_f <- predict(random_forest, data.frame(cbind(X_train)))
y_test_pred_f <- predict(random_forest, data.frame(cbind(X_test)))
y_validation_pred_f <- predict(random_forest, data.frame(cbind(X_validation)))

train_accuracy <- sum(y_train_pred_f == y_train) / length(y_train)
test_accuracy <- sum(y_test_pred_f == y_test) / length(y_test)
validation_accuracy <- sum(y_validation_pred_f == y_validation) / length(y_validation)

cat("Accuracy in Train:", train_accuracy, "\n")
cat("Accuracy in Test:", test_accuracy, "\n")
cat("Accuracy in Validation:", validation_accuracy, "\n")

y_pred <- predict(random_forest, data.frame(cbind(X_test)))
report <- confusionMatrix(y_pred, y_test)
print(report)

precision <- cvms(data.frame(cbind(y_usmp, X_usmp)), method = "prec", nfolds = 10, stratified = TRUE)
accuracy <- cvms(data.frame(cbind(y_usmp, X_usmp)), method = "class", nfolds = 10, stratified = TRUE)

cat("Mean precision:", mean(precision), "\n")
cat("Mean accuracy:", mean(accuracy), "\n")

cm <- as.table(table(y_test, y_test_pred_f))

heatmap.2(as.matrix(cm), Rowv = FALSE, Colv = FALSE, 
          col = brewer.pal(9, "YlOrRd"), 
          margins = c(3, 3),
          main = "Confusion Matrix Random Forest",
          xlab = "Predicted",
          ylab = "Actual")

parameter <- list(
  n_estimators = 36,
  max_depth = 9,
  min_child_weight = 1,
  learning_rate = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgboost_model <- xgboost(data = as.matrix(X_train), label = y_train, params = parameter, nrounds = 10, verbose = 1)

y_train_pred_xgb <- predict(xgboost_model, as.matrix(X_train))
y_test_pred_xgb <- predict(xgboost_model, as.matrix(X_test))
y_validation_pred_xgb <- predict(xgboost_model, as.matrix(X_validation))

y_train_pred_xgb <- ifelse(y_train_pred_xgb > 0.5, 1, 0)
y_test_pred_xgb <- ifelse(y_test_pred_xgb > 0.5, 1, 0)
y_validation_pred_xgb <- ifelse(y_validation_pred_xgb > 0.5, 1, 0)

train_accuracy <- accuracy(y_train_pred_xgb, y_train)
test_accuracy <- accuracy(y_test_pred_xgb, y_test)
validation_accuracy <- accuracy(y_validation_pred_xgb, y_validation)

cat("Accuracy in Train:", train_accuracy, "\n")
cat("Accuracy in Test:", test_accuracy, "\n")
cat("Accuracy in Validation:", validation_accuracy, "\n")

classification_report <- confusionMatrix(y_test_pred_xgb, y_test)
print(classification_report)

precision <- cvms(as.data.frame(cbind(y_usmp, X_usmp)), method = "prec", nfolds = 10, stratified = TRUE)
accuracy <- cvms(as.data.frame(cbind(y_usmp, X_usmp)), method = "class", nfolds = 10, stratified = TRUE)

cat("Mean precision:", mean(precision), "\n")
cat("Mean accuracy:", mean(accuracy), "\n")

cm <- confusionMatrix(y_test_pred_xgb, y_test)

heatmap.2(as.matrix(cm$confusionMatrix), Rowv = FALSE, Colv = FALSE, 
          col = colorRampPalette(c('white', 'red'))(100), 
          margins = c(3, 3),
          main = "Confusion Matrix XGBoost",
          xlab = "Predicted",
          ylab = "Actual")

random_forest <- randomForest(fraud ~ ., data = data.frame(cbind(y_train, X_train)), ntree = 10, mtry = ncol(X_train), nodesize = 2)

xgboost_model <- xgboost(data = as.matrix(X_train), label = y_train, nrounds = 10, verbose = 1)

ensemble_model <- caretList(
  models = list(random_forest, xgboost_model),
  data = data.frame(cbind(y_train, X_train)),
  trControl = trainControl(method = "cv", number = 5, savePredictions = "final")
)

ensemble_model <- caretEnsemble(ensemble_model, method = "ensemble")

y_pred <- predict(ensemble_model$finalModel, as.matrix(X_test))

accuracy <- sum(y_pred == y_test) / length(y_test)

cat("Accuracy:", accuracy, "\n")

classification_report <- confusionMatrix(y_pred, y_test)
print(classification_report)

precision <- cvms(as.data.frame(cbind(y_usmp, X_usmp)), method = "prec", nfolds = 10, stratified = TRUE)
accuracy <- cvms(as.data.frame(cbind(y_usmp, X_usmp)), method = "class", nfolds = 10, stratified = TRUE)

cat("Mean precision:", mean(precision), "\n")
cat("Mean accuracy:", mean(accuracy), "\n")

cm <- confusionMatrix(y_pred, y_test)

heatmap.2(as.matrix(cm$confusionMatrix), Rowv = FALSE, Colv = FALSE, 
          col = colorRampPalette(c('white', 'red'))(100), 
          margins = c(3, 3),
          main = "Confusion Matrix Ensemble Hard",
          xlab = "Predicted",
          ylab = "Actual")

random_forest <- randomForest(fraud ~ ., data = data.frame(cbind(y_train, X_train)), ntree = 10, mtry = ncol(X_train), nodesize = 2)

xgboost_model <- xgboost(data = as.matrix(X_train), label = y_train, nrounds = 10, verbose = 1)

ensemble_model <- caretList(
  models = list(random_forest, xgboost_model),
  data = data.frame(cbind(y_train, X_train)),
  trControl = trainControl(method = "cv", number = 5, savePredictions = "final")
)

ensemble_model <- caretEnsemble(ensemble_model, method = "ensemble", metric = "ROC")

y_pred <- predict(ensemble_model$finalModel, as.matrix(X_test))

y_pred <- ifelse(y_pred > 0.5, 1, 0)

accuracy <- sum(y_pred == y_test) / length(y_test)

cat("Accuracy:", accuracy, "\n")

classification_report <- confusionMatrix(y_pred, y_test)
print(classification_report)

precision <- cvms(as.data.frame(cbind(y_usmp, X_usmp)), method = "prec", nfolds = 10, stratified = TRUE)
accuracy <- cvms(as.data.frame(cbind(y_usmp, X_usmp)), method = "class", nfolds = 10, stratified = TRUE)

cat("Mean precision:", mean(precision), "\n")
cat("Mean accuracy:", mean(accuracy), "\n")

cm <- confusionMatrix(y_pred, y_test)

heatmap.2(as.matrix(cm$confusionMatrix), Rowv = FALSE, Colv = FALSE, 
          col = colorRampPalette(c('white', 'red'))(100), 
          margins = c(3, 3),
          main = "Confusion Matrix Ensemble Soft",
          xlab = "Predicted",
          ylab = "Actual")

