# Adam Nadoba
# Kaggle - Digit Recognizer
# SVM classifier used with PCA
# beware - it takes about 10 minutes on i7/16GB to get the results

setwd('/users/adam/studia/inteligencjaObl/projekt')

# download packages
install.packages('readr')
install.packages('e1071')

# load libraries
library(e1071)
library(readr)
library(scales)
library(class)

# for constant and repeatable results
set.seed(1) 

# define constant
NumComponents = 100

# load data
trainRaw <- read_csv("./train.csv")
testRaw <- read_csv("./test.csv")

labels <- as.factor(trainRaw[1:nrow(trainRaw),1])
trainWithoutLabels <- trainRaw[-1]

test <- testRaw

# apply PCA
pca <- prcomp(trainRaw[,-1])

trainPca <- predict(pca, newdata=trainWithoutLabels)[,1:NumComponents]
trainWithoutLabels <- as.data.frame(trainPca)

testPca <- predict(pca, newdata=test)[,1:NumComponents]
test <- as.data.frame(testPca)

# Support Vector Machines classifying

svmModel <- svm(x = trainWithoutLabels, y = labels)
svmPred <- predict(svmModel, test)

submission <- data.frame(ImageId=1:nrow(test), Label=levels(labels)[svmPred])
write_csv(as.data.frame(submission), "./SvmPcaTest.csv")
