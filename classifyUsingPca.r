# Adam Nadoba
# Kaggle - Digit Recognizer
# several classifying algorithms used with PCA

setwd('/users/adam/studia/inteligencjaObl/projekt')

# download packages
install.packages('neuralnet')
install.packages('randomForest')
install.packages('readr')
install.packages('e1071')

# load libraries
library(e1071)
library(neuralnet)
library(gmodels)
library(randomForest)
library(readr)
library(scales)
library(party)
library(class)

# for constant and repeatable results
set.seed(1) 

# define constants
NumTrain = 31000
NumTest = 11000
NumTrees = 25
NumComponents = 25

# load data
trainRaw <- read_csv("./train.csv")

totalRows <- nrow(trainRaw)

#rows <- sample(1:nrow(train), numTrain)
rows <- totalRows:(totalRows-NumTrain)

labels <- as.factor(trainRaw[rows,1])
trainWithoutLabels <- trainRaw[rows,-1]
trainWithLabels <- trainRaw[rows,]

test <- head(trainRaw[-1], NumTest)
testLabels <- as.factor(trainRaw[1:NumTest, 1])

# apply PCA
pca <- prcomp(trainRaw[,-1])

trainPca <- predict(pca, newdata=trainWithoutLabels)[,1:NumComponents]
trainWithoutLabels <- as.data.frame(trainPca)

trainWithLabels <- as.data.frame(trainPca)
trainWithLabels$label <- labels

trainWithLabelsMatrix <- cbind(trainPca, 'label' = trainRaw[rows,1])

testPca <- predict(pca, newdata=test)[,1:NumComponents]
test <- as.data.frame(testPca)

# universal formula to work on data PCs

formulaString <- paste('label ~ ', paste(paste('PC', 1:NumComponents, sep=''), collapse='+'), sep='')
formula <- as.formula(formulaString)

# precision function to verify the results
precision <- function(correct, predicted) {
  legitimate = 0
  total = nrow(predicted)
  
  for (i in 1:total) {
    if (correct[i,1]==predicted[i,1]) {
      legitimate = legitimate + 1
    }
  }
  
  return(percent(legitimate/total))
}


# actual Random Forest classifying
print("--------------------------------------------------")
print("Random Forest")

rf <- randomForest(x=trainWithoutLabels, y=labels, xtest=test, ytest=testLabels, ntree=NumTrees)
randomForestPredictions <- data.frame(ImageId=1:nrow(test), Label=levels(labels)[rf$test$predicted])
randomForestPrecision <- precision(trainRaw[1], randomForestPredictions[2])

sprintf("Precision: %s", randomForestPrecision)
print("Confusion matrix: ")
print(rf$test$confusion)


# k Nearest Neighbours classifying
NeighbourCount = 5
print("--------------------------------------------------")
sprintf("%s Nearest Neighbours", NeighbourCount)

nn <- knn(train = trainWithoutLabels, cl = labels, test = test, k = NeighbourCount)
nnPredictions <- as.data.frame(nn)
nnPrecision <- precision(trainRaw[1], nnPredictions)

sprintf("Precision: %s", nnPrecision)
confusionMatrix <- table(testLabels, as.factor(nnPredictions[,1]), dnn = c("Actual", "Predicted"))
print("Confusion matrix: ")
print(confusionMatrix)


# neural net classifying
#HiddenNeuronCount = 15
#
#net <- neuralnet(formula, data = trainWithLabelsMatrix, hidden = HiddenNeuronCount)
#results <- compute(net, test)$net.result
#netPrecision <- precision(trainRaw[1], results)

#print("--------------------------------------------------")
#sprintf("Neural Net (hidden = %s)", HiddenNeuronCount)
#sprintf("Precision: %s", netPrecision)
#print("Confusion matrix: ")
#confusion <- table(testLabels, results, dnn = c("Actual", "Predicted"))
#print(confusion)


# Conditional Inference Trees classifying
print("--------------------------------------------------")
print("Conditional Inference Trees")

digitsTree <- ctree(formula, data = trainWithLabels)
ctreePred <- predict(digitsTree, test)
ctreePrecision <- precision(trainRaw[1], as.data.frame(ctreePred))

sprintf("Precision: %s", ctreePrecision)
confusion <- table(as.factor(testLabels), ctreePred, dnn = c("Actual", "Predicted"))
print("Confusion matrix: ")
print(confusion)


# Naive Bayes classifying
print("--------------------------------------------------")
print("Naive Bayes")

nb <- naiveBayes(x = trainWithoutLabels, y = labels)
nbPred <- predict(nb, test)
nbPrecision <- precision(trainRaw[1], as.data.frame(nbPred))

sprintf("Precision: %s", nbPrecision)
nbConfusion <- table(testLabels, nbPred, dnn = c("Actual", "Predicted"))
print("Confusion matrix: ")
print(nbConfusion)


# Support Vector Machines classifying
print("--------------------------------------------------")
print("Support Vector Machines")

svmModel <- svm(x = trainWithoutLabels, y = labels)
svmPred <- predict(svmModel, test)
svmPrecision <- precision(trainRaw[1], as.data.frame(svmPred))
svmPrecision

sprintf("Precision: %s", svmPrecision)
svmConfusion <- table(testLabels, svmPred, dnn = c("Actual", "Predicted"))
print("Confusion matrix: ")
print(svmConfusion)
