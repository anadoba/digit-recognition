# Adam Nadoba

setwd('/users/adam/studia/inteligencjaObl/projekt')

# download packages
install.packages('randomForest')
install.packages('readr')

# load libraries
library(randomForest)
library(readr)
library(scales)

# for constant and repeatable results
set.seed(1) 

# define constants
NumTrain = 31000
NumTest = 11000
NumTrees = 25

# load data
trainRaw <- read_csv("./train.csv")

totalRows <- nrow(trainRaw)

#rows <- sample(1:nrow(train), numTrain)
rows <- totalRows:(totalRows-NumTrain)
labels <- as.factor(trainRaw[rows,1])
trainWithoutLabels <- trainRaw[rows,-1]

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

rf <- randomForest(x=trainWithoutLabels, y=as.factor(labels), xtest=test, ytest=testLabels, ntree=NumTrees)
randomForestPredictions <- data.frame(ImageId=1:nrow(test), Label=levels(labels)[rf$test$predicted])

randomForestPrecision <- precision(trainRaw[1], randomForestPredictions[2])

print("--------------------------------------------------")
print("Random Forest")
sprintf("Precision: %s", randomForestPrecision)
print("Confusion matrix: ")
print(rf$test$confusion)
