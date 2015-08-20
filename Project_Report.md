# Practical Machine Learning - Project Report
Raeger Tay  
20 August 2015  

# Dataset
[Training Data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)  
[Test Data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)  
[Data Reference](http://groupware.les.inf.puc-rio.br/har)  

# Data Cleaning
The training dataset is first loaded, and empty cells are filled with NA. Columns 1 to 7 contains participants' names, timestamp and window identifier. These are metadata and therefore are removed. 

```r
training <- read.csv(file = "pml-training.csv", na.strings = c("", "NA"))
names(training)[1:7]
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
## [7] "num_window"
```

```r
training <- training[, -(1:7)]
```

<br>
Next, it is noted that there are multiple columns with high percentage of NAs. These columns contain summary statistics like kurtosis and skewness for each window. These columns are also removed. This leaves us with 19,622 observations and 52 variables. The 53th column contains the outcome.

```r
na.count <- sapply(training, function(x) {sum(is.na(x))/nrow(training)})
training <- training[, na.count == 0] # Remove summary statistics
dim(training)
```

```
## [1] 19622    53
```

<br>
The training set is partitioned into a sub-training set and a sub-test set on a 70:30 split. The model will be trained using the sub-training set and then test on the sub-test set.

```r
library(caret)
inTrain <- createDataPartition(training$classe, p = 0.7, list = FALSE)
sub.training <- training[inTrain, ]
sub.test <- training[-inTrain, ]
```
 
# Prediction Model
The Breiman's random forest algorithm is used for the training of our model. It can be called using the function randomForest() from the "randomForest" package. Using the trained model, the outcome is predicted on the sub.test dataset. The confusion matrix is shown. 

```r
library(randomForest)
modelFit <- randomForest(classe ~ ., data = sub.training)
sub.test.pred <- predict(modelFit, sub.test[,-53])
c <- confusionMatrix(sub.test.pred, sub.test$classe)
c
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    5    0    0    0
##          B    0 1131    4    0    0
##          C    0    3 1021   15    1
##          D    0    0    1  947    3
##          E    0    0    0    2 1078
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9942         
##                  95% CI : (0.9919, 0.996)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9927         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9930   0.9951   0.9824   0.9963
## Specificity            0.9988   0.9992   0.9961   0.9992   0.9996
## Pos Pred Value         0.9970   0.9965   0.9817   0.9958   0.9981
## Neg Pred Value         1.0000   0.9983   0.9990   0.9966   0.9992
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1922   0.1735   0.1609   0.1832
## Detection Prevalence   0.2853   0.1929   0.1767   0.1616   0.1835
## Balanced Accuracy      0.9994   0.9961   0.9956   0.9908   0.9979
```
The accuracy predicted on the sub.test set is 0.994. The out-of-sample error is then 0.00578. It is noted that no cross-validation is required for random forest as it is already built inside the algorithm.

# Prediction Results
Now, given the good accuracy on our out-of-sample dataset, the model will be used to predict the answers for the test set. The test set is subjected to the same cleaning process as our training set.

```r
test <- read.csv(file = "pml-testing.csv", na.strings = c("", "NA"))
test <- test[, -(1:7)] # Remove meta-data
na.count <- sapply(test, function(x) {sum(is.na(x))/nrow(test)})
test <- test[, na.count == 0] # Remove summary statistics
test.pred <- predict(modelFit, test[,-53])
as.character(test.pred)
```

```
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```

The predictions scored a result of 20/20. 
