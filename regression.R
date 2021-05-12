y24 <- read.table("data/target_variables/fluorination_y24.txt",row.names=1,nrows=24)

descriptor <- read.table("descriptor.txt", header=TRUE)

library(glmnet)

EN <- cv.glmnet(as.matrix(descriptor), as.matrix(y24), nfold=24, alpha=0.9, grouped=FALSE) 
s = EN$lambda.min
coefficient = coef(EN, s)

write.table(as.matrix(coefficient), file="output/coefficient.txt", col.name=FALSE, quote=FALSE)