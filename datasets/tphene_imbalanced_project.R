# ========================================
# Multiple Hypothesis Testing
# Part 1: K-fold Cross-Validation Paired t-Test
# Part 2: Analysis of Variance (ANOVA) Test
# Part 3: Wilcoxon Signed Rank test
# ========================================
# Load the required R packages
set.seed(1000)
require(C50)
require(caret)
require(kernlab)
require(e1071)
require(ROCR)
#all data loaded
Iris_data = read.csv("datasets/Iris_data.txt",header = FALSE)
Ecoli_data = read.csv("datasets/Ecoli_data.csv",header = FALSE)
Glass_data = read.csv("datasets/Glass_data.txt",header = FALSE)
WBC_data = read.csv("datasets/Wisconsin_Breast_Cancer_data.txt",header = FALSE)
Yeast_data = read.csv("datasets/Yeast_data.csv",header = FALSE)
Glass_data$V11 <- as.factor(Glass_data$V11)

# **********************************************
# Part 1: K-fold Cross-Validation Paired t-Test
# *****************************************
#question 1
# Load the iris data set
Iris_data_shuffle <- Iris_data[sample(nrow(Iris_data)),]
folds <- cut(seq(1,nrow(Iris_data_shuffle)),breaks=10,labels=FALSE)
c5error = c(0,0,0,0,0,0,0,0,0,0)
svmError =  c(0,0,0,0,0,0,0,0,0,0)
# Randomize the data and perform 10-fold Cross-Validation
# See ?sample and ?cvFolds


# Use the training set to train a C5.0 decision tree and Support Vector Machine


# Make predictions on the test set and calculate the error percentages made by both the trained models



for(i in 1:10){
  #Segment your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- Iris_data_shuffle[testIndexes,]
  trainData <- Iris_data_shuffle[-testIndexes,]
  c5model = C5.0(trainData[,-5],trainData[,5])
  p = predict(c5model,testData[,-5])
  t = table(testData[,5],Predicted = p)
  total_correct_classified_c5 = t[1] + t[5] + t[9]
  print(total_correct_classified_c5)
  c5error[i] = (15 - total_correct_classified_c5)/15*100
  
  
  svm_model = ksvm(trainData$V5 ~ .,trainData)
  svmPred = predict(svm_model,testData[,-5])
  t1 = table(testData[,5],svmPred)
  total_correct_classified_svm = t1[1] + t1[5] + t1[9]
  print(total_correct_classified_svm)
  svmError[i] = (15 - total_correct_classified_svm)/15*100
  
  #svmError = c(svmError, sum(svmPred == testData$Iris.setosa) * 100 / length(svmPred))
}
# Perform K-fold Cross-Validation Paired t-Test to compare the means of the two error percentages
c5error
svmError
t.test(c5error,svmError,paired = TRUE)
# *****************************************
# Part 2: Analysis of Variance (ANOVA) Test
# *****************************************

# Load the Breast Cancer data set 

# Randomize the data and perform 10-fold Cross-Validation
# See ?sample and ?cvFolds

#question 2
WBC_data_shuffle <- WBC_data[sample(nrow(WBC_data)),]
folds <- cut(seq(1,nrow(WBC_data_shuffle)),breaks=10,labels=FALSE)
nberror_wbc =  c(0,0,0,0,0,0,0,0,0,0)
lrerror_wbc =  c(0,0,0,0,0,0,0,0,0,0)
c5error_wbc = c(0,0,0,0,0,0,0,0,0,0)
svmError_wbc = c(0,0,0,0,0,0,0,0,0,0)

# Use the training set to train following classifier algorithms
# 	1. C5.0 decision tree (see ?C5.0 in C50 package)
# 	2. Support Vector Machine (see ?ksvm in kernlab package)
# 	3. Naive Bayes	(?naiveBayes in e1071 package) 
# 	4. Logistic Regression (?glm in stats package) 


#i=1 
for(i in 1:10){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- WBC_data_shuffle[testIndexes,]
  trainData <- WBC_data_shuffle[-testIndexes,]
  c5model = C5.0(trainData[,-2],trainData[,2])
  p = predict(c5model,testData[,-2])
  t = table(testData[,2],Predicted = p)
  total_correct_classified_c5 = t[1] + t[4]
  print(total_correct_classified_c5)
  c5error_wbc[i] = (57 - total_correct_classified_c5)/57*100
  
  svm_model = ksvm(trainData$V2 ~ .,trainData)
  svmPred = predict(svm_model,testData[,-2])
  t1 = table(testData[,2],svmPred)
  total_correct_classified_svm = t1[1] + t1[4]
  print(total_correct_classified_svm)
  svmError_wbc[i] = (57 - total_correct_classified_svm)/57*100
  
  nbmodel = naiveBayes(trainData$V2~.,trainData)
  nbpred = predict(nbmodel,testData[,-2])
  t2 = table(testData[,2],nbpred)
  total_correct_classified_nb = t[1] + t[4]
  nberror_wbc[i] = (57 - total_correct_classified_nb)/57*100
  
  lrmodel = glm(formula = trainData$V2~.,data = trainData,family = binomial(link = 'logit'))
  lrpred = predict(lrmodel,testData[,-2],type="response")
  logRegPred = rep('B', dim(testData[,-2])[1])
  logRegPred[lrpred > 0.5] = 'M'
  t3 = table(testData[,2],logRegPred)
  total_correct_classified_lr = t3[1]+t3[4]
  lrerror_wbc[i] = (57 - total_correct_classified_lr)/57*100
 # pr <- prediction(lrpred, testData$V2)
#  auc <- performance(pr, measure = "auc")
#  auc <- auc@y.values[[1]]
#   lrerror[i]= (1-auc)*100
} 
  #logRegModel <- glm(formula = V2 ~ ., family = binomial(link = 'logit'), data = trainData)
  #logRegProb <- predict(logRegModel, testData[,-2], type = "response")
  #logRegPred = rep('B', dim(testData[,-2])[1])
  #logRegPred[logRegProb > 0.5] = 'M'
  Total_error = c(c5error_wbc,svmError_wbc,nberror_wbc,lrerror_wbc)
  classc5 = rep("c5", 10)
  classsvm = rep("svm", 10)
  classnb = rep("nb", 10)
  classlr = rep("lr", 10)
  
  classifiers = c(classc5,classsvm,classnb,classlr)
  error = data.frame(Total_error,classifiers)
  # Make predictions on the test set and calculate the error percentages made by the trained models
  
  # Compare the performance of the different classifiers using ANOVA test (see ?aov)
  aov_model = aov(Total_error ~ classifiers, data = error)
  aov_model
  summary(aov_model)
  
#Question 3
  
  
  # *****************************************
  # Part 3: Wilcoxon Signed Rank test
  # *****************************************
  # Load the following data sets,
  # 1. Iris 
  
  
  # 2. Ecoli 
  
  
  # 3. Wisconsin Breast Cancer
  
  
  # 4. Glass
  
  
  # 5. Yeast
Iris_data_shuffle <- Iris_data[sample(nrow(Iris_data)),]
folds_iris <- cut(seq(1,nrow(Iris_data_shuffle)),breaks=10,labels=FALSE)
c5erroriris = c(0,0,0,0,0,0,0,0,0,0)
svmErroriris =  c(0,0,0,0,0,0,0,0,0,0)

Ecoli_data_shuffle <- Ecoli_data[sample(nrow(Ecoli_data)),]
folds_Ecoli <- cut(seq(1,nrow(Ecoli_data_shuffle)),breaks=10,labels=FALSE)
c5errorecoli = c(0,0,0,0,0,0,0,0,0,0)
svmErrorecoli =  c(0,0,0,0,0,0,0,0,0,0)

Glass_data_shuffle <- Glass_data[sample(nrow(Glass_data)),]
folds_Glass <- cut(seq(1,nrow(Glass_data_shuffle)),breaks=10,labels=FALSE)
c5errorglass = c(0,0,0,0,0,0,0,0,0,0)
svmErrorglass =  c(0,0,0,0,0,0,0,0,0,0)

WBC_data_shuffle <- WBC_data[sample(nrow(WBC_data)),]
folds_WBC <- cut(seq(1,nrow(WBC_data_shuffle)),breaks=10,labels=FALSE)
c5errorwbc = c(0,0,0,0,0,0,0,0,0,0)
svmErrorwbc =  c(0,0,0,0,0,0,0,0,0,0)

Yeast_data_shuffle <- Yeast_data[sample(nrow(Yeast_data)),]
folds_Yeast <- cut(seq(1,nrow(Yeast_data_shuffle)),breaks=10,labels=FALSE)
c5erroryeast = c(0,0,0,0,0,0,0,0,0,0)
svmErroryeast =  c(0,0,0,0,0,0,0,0,0,0)
# Randomize the data and perform 10-fold Cross-Validation
# See ?sample and ?cvFolds
# Use the training set to train following classifier algorithms
# 	1. C5.0 decision tree (see ?C5.0 in C50 package)
# 	2. Support Vector Machine (see ?ksvm in kernlab package)
for(i in 1:10){
  #iris data
  #c5
  testIndexes_iris <- which(folds_iris==i,arr.ind=TRUE)
  testData_iris <- Iris_data_shuffle[testIndexes_iris,]
  trainData_iris <- Iris_data_shuffle[-testIndexes_iris,]
  c5model_iris = C5.0(trainData_iris[,-5],trainData_iris[,5])
  p_iris = predict(c5model_iris,testData_iris[,-5])
  t_iris = table(testData_iris[,5],Predicted = p_iris)
  total_correct_classified_c5_iris = t_iris[1] + t_iris[5] + t_iris[9]
  #print(total_correct_classified_c5)
  c5erroriris[i] = (15 - total_correct_classified_c5_iris)/15*100
  #svm
  svm_model_iris = ksvm(trainData_iris$V5 ~ .,trainData_iris)
  svmPred_iris = predict(svm_model_iris,testData_iris[,-5])
  t1_iris = table(testData_iris[,5],svmPred_iris)
  total_correct_classified_svm_iris = t1_iris[1] + t1_iris[5] + t1_iris[9]
  #print(total_correct_classified_svm_iris)
  svmErroriris[i] = (15 - total_correct_classified_svm_iris)/15*100
  #glass data
  #c5
  testIndexes_glass <- which(folds_Glass==i,arr.ind=TRUE)
  testData_glass <- Glass_data_shuffle[testIndexes_glass,]
  trainData_glass <- Glass_data_shuffle[-testIndexes_glass,]
  c5model_glass = C5.0(trainData_glass[,-11],(trainData_glass[,11]))
  p_glass = predict(c5model_glass,testData_glass[,-11])
  t_glass = confusionMatrix(testData_glass[,11],p_glass)
  total_correct_classified_c5_glass = sum(diag(t_glass$table))
  print(total_correct_classified_c5_glass)
  c5errorglass[i] = (21 - total_correct_classified_c5_glass)/21*100
  #svm
  svm_model_glass = ksvm(trainData_glass$V11 ~ .,(trainData_glass))
  svmPred_glass = predict(svm_model_glass,testData_glass[,-11])
  t1_glass = confusionMatrix(testData_glass[,11],svmPred_glass)
  total_correct_classified_svm_glass = sum(diag(t1_glass$table))
  print(total_correct_classified_svm_glass)
  svmErrorglass[i] = (21 - total_correct_classified_svm_glass)/21*100
  
  #yeast data
  #c5
  testIndexes_yeast <- which(folds_Yeast==i,arr.ind=TRUE)
  testData_yeast <- Yeast_data_shuffle[testIndexes_yeast,]
  trainData_yeast<- Yeast_data_shuffle[-testIndexes_yeast,]
  c5model_yeast = C5.0(trainData_yeast[,-10],trainData_yeast[,10])
  p_yeast = predict(c5model_yeast,testData_yeast[,-10])
  t_yeast = confusionMatrix(testData_yeast[,10],p_yeast)
  total_correct_classified_c5_yeast = sum(diag(t_yeast$table))
  print(total_correct_classified_c5_yeast)
  c5erroryeast[i] = (148 - total_correct_classified_c5_yeast)/148*100
  #svm
  svm_model_yeast = ksvm(trainData_yeast$V10 ~ .,trainData_yeast)
  svmPred_yeast = predict(svm_model_yeast,testData_yeast[,-10])
  t1_yeast = confusionMatrix(testData_yeast[,10],svmPred_yeast)
  total_correct_classified_svm_yeast = sum(diag(t1_yeast$table))
  print(total_correct_classified_svm_yeast)
  svmErroryeast[i] = (148 - total_correct_classified_svm_yeast)/148*100
  
  #ecoli data
  #c5
  testIndexes_ecoli <- which(folds_Ecoli==i,arr.ind=TRUE)
  testData_ecoli <- Ecoli_data_shuffle[testIndexes_ecoli,]
  trainData_ecoli <- Ecoli_data_shuffle[-testIndexes_ecoli,]
  c5model_ecoli = C5.0(trainData_ecoli[,-9],trainData_ecoli[,9])
  p_ecoli = predict(c5model_ecoli,testData_ecoli[,-9])
  t_ecoli = confusionMatrix(testData_ecoli[,9],p_ecoli)
  total_correct_classified_c5_ecoli = sum(diag(t_ecoli$table))
  print(total_correct_classified_c5_ecoli)
  c5errorecoli[i] = (34 - total_correct_classified_c5_ecoli)/34*100
  #svm
  svm_model_ecoli = ksvm(trainData_ecoli$V9 ~ .,trainData_ecoli)
  svmPred_ecoli = predict(svm_model_ecoli,testData_ecoli[,-9])
  t1_ecoli = confusionMatrix(testData_ecoli[,9],svmPred_ecoli)
  total_correct_classified_svm_ecoli = sum(diag(t1_ecoli$table))
  print(total_correct_classified_svm_ecoli)
  svmErrorecoli[i] = (34 - total_correct_classified_svm_ecoli)/34*100
}
# Make predictions on the test set and calculate the error percentages made by the trained models

# Compare the performance of the different classifiers using Wilcoxon Signed Rank test (see ?wilcox.test)
final_c5_iris = mean(c5erroriris)
final_svm_iris = mean(svmErroriris)
final_c5_wbc = mean(c5error_wbc)
final_svm_wbc = mean(svmError_wbc)
final_c5_ecoli = mean(c5errorecoli)
final_svm_ecoli = mean(svmErrorecoli)
final_c5_yeast = mean(c5erroryeast)
final_svm_yeast = mean(svmErroryeast)
final_c5_glass = mean(c5errorglass)
final_svm_glass = mean(svmErrorglass)
errorc5 <- rbind(c(final_c5_iris), c(final_c5_wbc), c(final_c5_ecoli), c(final_c5_glass), c(final_c5_yeast))
errorsvm <- rbind(c(final_svm_iris), c(final_svm_wbc), c(final_svm_ecoli), c(final_svm_glass), c(final_svm_yeast))
errorc5 <- cbind(errorc5, "C5.0")
errorsvm <- cbind(errorsvm, "SVM")
errors.df <- rbind(errorc5, errorsvm)
colnames(errors.df) <- c("errors", "classifiers")
errors.df <- as.data.frame(errors.df)
errors.df$errors <- as.numeric(errors.df$errors)
wilcox.test(formula = errors ~ classifiers, data = errors.df, paired = TRUE)
