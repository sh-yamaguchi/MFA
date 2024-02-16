mfa <- function(x,xtest,ym,alp,pmax,outcomes,directory){
  n = nrow(ym)
  library(glmnet)
  if (alp == 1.0){
    if (pmax == "off"){
    lasso <- cv.glmnet(as.matrix(x),as.matrix(ym),nfold=n,alpha=alp,grouped=FALSE)
    }
    else{
    lasso <- cv.glmnet(as.matrix(x),as.matrix(ym),nfold=n,alpha=alp,pmax=n,grouped=FALSE)
    }
  }
  else {
    lasso <- cv.glmnet(as.matrix(x),as.matrix(ym),nfold=n,alpha=alp,grouped=FALSE)
  }
  ymean <- mean(as.matrix(ym))
  s = lasso$lambda.min
  vyploo <- c(seq(n))
  vyp <- c(seq(n))
  for (i in 1:n) {
    if (alp == 1.0){
      if (pmax == "off"){
      cvlasso <- glmnet(as.matrix(x[-i,]),as.matrix(ym[-i,]),alpha=alp) 
      }
      else{
      cvlasso <- glmnet(as.matrix(x[-i,]),as.matrix(ym[-i,]),alpha=alp,pmax=n)  
      }
    }
    else {
      cvlasso <- glmnet(as.matrix(x[-i,]),as.matrix(ym[-i,]),alpha=alp)
    }
    yploo <- predict(cvlasso,as.matrix(x[i,]),s)
    vyploo[i] <- yploo
    }
  vyp <- c(predict(lasso,as.matrix(x),s))
  vyp_test <- c(predict(lasso,as.matrix(xtest),s))
  vyp <- t(vyp)
  vyploo <- c(vyploo)
  ym <- t(ym)
  output <- rbind(ym,vyp,vyploo)
  output <- t(output)
  colnames(output) <- c("measured","pred","pred_LOOCV")
  output_name <- paste(directory, outcomes,"_output.csv",sep="")
  output_name_test <- paste(directory, outcomes,"_output_test.csv",sep="")
  write.csv(output, output_name, quote=F,row.names = T)
  write.csv(vyp_test, output_name_test, quote=F)
  coefficient = coef(lasso,s)
  coefficient_name <- paste(directory,"coefficient_",outcomes,".txt",sep="")
  write.table(as.matrix(coefficient),file=coefficient_name,col.name=FALSE,quote=FALSE)
}
