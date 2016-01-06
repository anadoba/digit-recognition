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
NumTrain = 30000
NumTest = 10000
NumTrees = 25

# load data
trainRaw <- read_csv("./train.csv")

numRows <- nrow(trainRaw)

#rows <- sample(1:nrow(train), numTrain)
rows <- numRows:(numRows-NumTrain)
labels <- as.factor(trainRaw[rows,1])
trainWithoutLabels <- trainRaw[rows,-1]

test <- head(trainRaw[-1], NumTest)

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

rf <- randomForest(x=trainWithoutLabels, y=as.factor(labels), xtest=test, ntree=NumTrees)
randomForestPredictions <- data.frame(ImageId=1:nrow(test), Label=levels(labels)[rf$test$predicted])

randomForestPrecision <- precision(train[1], randomForestPredictions[2])

sprintf("Random Forest precision: %s", randomForestPrecision)
