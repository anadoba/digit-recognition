# Adam Nadoba

setwd('/users/adam/studia/inteligencjaObl/projekt')

# download packages
install.packages("neuralnet")
install.packages('randomForest')
install.packages('readr')

# load libraries
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
NumTrain = 310
NumTest = 110
NumTrees = 25

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

rf <- randomForest(x=trainWithoutLabels, y=labels, xtest=test, ytest=testLabels, ntree=NumTrees)
randomForestPredictions <- data.frame(ImageId=1:nrow(test), Label=levels(labels)[rf$test$predicted])

randomForestPrecision <- precision(trainRaw[1], randomForestPredictions[2])

print("--------------------------------------------------")
print("Random Forest")
sprintf("Precision: %s", randomForestPrecision)
print("Confusion matrix: ")
print(rf$test$confusion)


# k Nearest Neighbours classifying
NeighbourCount = 5

nn <- knn(train = trainWithoutLabels, cl = labels, test = test, k = NeighbourCount)
nnPredictions <- as.data.frame(nn)
nnPrecision <- precision(trainRaw[1], nnPredictions)

print("--------------------------------------------------")
sprintf("%s Nearest Neighbours", NeighbourCount)
sprintf("Precision: %s", nnPrecision)
print("Confusion matrix: ")
confusionMatrix <- table(testLabels, as.factor(nnPredictions[,1]), dnn = c("Actual", "Predicted"))
print(confusionMatrix)


# neural net classifying
HiddenNeuronCount = 15

formulaString <- paste('label ~ ', paste(paste('pixel', 0:783, sep=''), collapse='+'), sep='')
formula <- as.formula(formulaString)

net <- neuralnet(formula, data = trainWithLabels, hidden = HiddenNeuronCount)
results <- compute(net, test)$net.result
results <- round(results)
netPrecision <- precision(trainRaw[1], results)

print("--------------------------------------------------")
sprintf("Neural Net (hidden = %s)", HiddenNeuronCount)
sprintf("Precision: %s", netPrecision)
print("Confusion matrix: ")
confusion <- table(testLabels, results, dnn = c("Actual", "Predicted"))
print(confusion)

# ctree classifying
digitsTree <- ctree(formula, data = trainWithLabels)
results <- predict(digitsTree, test)
results <- round(results)
ctreePrecision <- precision(trainRaw[1], results)

print("--------------------------------------------------")
print("Conditional Inference Trees")
sprintf("Precision: %s", ctreePrecision)
print("Confusion matrix: ")
confusion <- table(testLabels, results, dnn = c("Actual", "Predicted"))
print(confusion)
