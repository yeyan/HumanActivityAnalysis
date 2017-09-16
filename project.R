library(tidyverse)
library(caret)

raw.data <- read_csv('pml-training.csv') %>%
  mutate(user_name=factor(user_name), classe=as.factor(classe))

# Remove na features
fields <- Reduce(rbind, lapply(colnames(raw.data),
                               function(name) data.frame(
                                 name=name,
                                 rate=mean(is.na(raw.data[[name]])))
                               ))
features <- fields %>%
  filter(rate < .1) %>%
  filter(!grepl('time', name)) %>%
  filter(!grepl('X1', name))

data = select(raw.data, features$name, -new_window)

inTrain <- createDataPartition(data$classe, p=.5, list=F)

training = data[inTrain,]
testing = data[-inTrain,]

prepro <- preProcess(training, method=c('medianImpute', 'center', 'scale'), thresh=.8) 
training.clean = predict(prepro, training)

library(e1071)

m1 <- svm(classe ~ ., training.clean)
m2 <- train(classe ~ ., training, method='knn', preProcess=c('center', 'scale'), na.action = na.omit)

model.acc <- function(x, y) {
  1/length(y) * sum(y == x)
}

# Optimistic accuracy
model.acc(predict(m1, training.clean), training.clean$classe)

# Out stample accuarcy
testing.clean = predict(prepro, testing)
model.acc(predict(m1, testing.clean), testing.clean$classe)

validating <- read_csv('pml-testing.csv') %>%
  select(features$name, -new_window) %>%
  mutate(user_name=factor(user_name))

validating = predict(prepro, validating)

predict(m1, validating)
