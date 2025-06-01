mfa_rf <- function(x, xtest, ym, outcomes, directory) {
  n <- nrow(ym)
  library(randomForest)
  
  set.seed(42)
  rf_model <- randomForest(x = as.matrix(x), y = as.matrix(ym))
  
  vyploo <- numeric(n)
  for (i in 1:n) {
    x_train <- x[-i, , drop = FALSE]
    y_train <- ym[-i, , drop = FALSE]
    rf_cv <- randomForest(x = x_train, y = as.matrix(y_train))
    vyploo[i] <- predict(rf_cv, x[i, , drop = FALSE])
  }

  vyp <- predict(rf_model, x)

  vyp_test <- predict(rf_model, as.matrix(xtest))

  output <- cbind(measured = ym, pred = vyp, pred_LOOCV = vyploo)
  output_name <- paste0(directory, outcomes, "output.csv")
  output_name_test <- paste0(directory, outcomes, "output_test.csv")
  write.csv(output, output_name, quote = FALSE, row.names = TRUE)
  write.csv(vyp_test, output_name_test, quote = FALSE)
}
