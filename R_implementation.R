rm(list=ls())
if(!require(randomForest)){
  install.packages("randomForest")
  library(randomForest)
}

# In this template random subsampling validation is used to predict
# an entire distribution of some response variable of new observations, thereby providing a generic way to
# compute a prediction interval. This method can be combined with any (machine learning) algorithm. This method relies 
# on the assumption that (i) new realizations of the modelled response variable are normally distributed around the predicted
# values of the main model and that (ii) the standard devation from this distribution can be estimated by using the 
# available features in the data. 

# For the purpose of this template, our aim is to predict the alcohol percentage in white wines. 
# Therefore, we will specify the response variable to be "alcohol". The other variables in this data 
# are used as features that we will use to predict the alcohol percentage. 
# This data is available on https://archive.ics.uci.edu/ml/datasets/wine 

path               <- "./data/winequality-white.csv"
data               <- read.csv(path, sep=";")
response           <- "alcohol"
n                  <- nrow(data)

# Ultimately, we want to be able to predict the alcohol percentage for any new wine, based on a set of features and a some
# statistical model that is trained on a sample of wines for which the alcohol percentage is already known. In order to mimic 
# this objective, we split our data in a train and a test set. You can specify the size of your test set in the 
# 'size_test' parameter. In this template, we set this size equal to 10% of the full data set. 


size_test          <- round(n/10,0)
ind_test           <- sample(1:n,size_test,replace = FALSE)
train              <- data[-ind_test,]
test               <- data[ind_test,]

# Each prediction interval has an associated confidence level that roughly quantifies the desired probability 
# with which the response value of a new observation lies in the calculated prediction interval. In this template we use a 
#confidence level of 95%. 

confidence         <- 0.95

# The parameter C defines the number of rounds of random subsampling validation that is used to calculate the residuals 
# for our error model. The parameter Q defines the size of the validation set that is used to validate the main model in 
# each subsampling validation replication.
# We also specify the two empty vectors in which we can save (i) the squared residuals and the (ii) indices corresponding 
# to each residual. 

Q                  <- round(nrow(train)/10,0)
C                  <- 20 

residuals          <- c()
saved_data_indices <- c()

# Now we will use the random subsampling validation to obtain the vector of squared residuals, so that we can quantify how far
# observations will approximately deviate from their predicted values. We will eventually use this vector of residuals 
# to esimate an observation-specific standard devation 


for (i in 1:C) {

  cv                  <- sample(1:nrow(train), Q, replace = FALSE)
  fit.cv              <- randomForest(as.formula(paste(response,"~.")), data = train[-cv,])
  
  pred.cv             <- predict(fit.cv,train[cv,!names(train) %in% response]) 
  
  e2_hat              <- (pred.cv - train[cv,response])^2
  residuals           <- c(residuals, e2_hat)
  saved_data_indices  <- c(saved_data_indices, cv)

}


# We are now going to use all the squared residuals and the indices that we saved to build a model that can predict how
# far a prediction of our main statistical model deviates from its true value, based on the set of features in our data.
# We use a random forest model to predict the squared residuals, because this algorithm often achieves good accuracy and 
# requires little to no parameter tuning. However, in principle any other model could be used as well. 

sd_estimator           <- randomForest(y=residuals, x=train[saved_data_indices,!names(train) %in% response])

# Now it is time to fit our main model. We fit a random forest model on our full training data. 
# Afterwards, we use our main model to predict the alcohol percentage of the observations in our test set. 

rf_main                <- randomForest(as.formula(paste(response,"~.")), data=train)
pred                   <- predict(rf_main,newdata=test[,!names(test) %in% response])

# We will now use our sd_estimator model to predict the deviation from the predicted value 
# for each observation in order to obtain an observation-specific approximated standard deviation.  

std                    <- (predict(sd_estimator,newdata=test[,!names(test) %in% response]))^0.5

# In order to calculate a prediction interval for each new observation, we use the approximated standard deviation for
# each observation in the test set and the assumption that the true realizations of our response variable (alcohol
# percentage) are normally distributed around the predicted values obtained from our main model. 

up <- pred + qt(1 - (1-confidence)/2,df=n-1)*std
lo <- pred - qt(1 - (1-confidence)/2,df=n-1)*std

# Now we have obtained our desired result. That is, for each observation in our test set we have calculated a
# predicted value (pred), and an upper (up) and lower (lo) bound of the 95% prediction interval.
# In order to check the validity of our prediction interval, we will calculate what percentage of the observations
# in the test set fall in their calculated prediction interval.  

perc_observations_inside_interval <- length(which(up>test[,response] & test[,response] > lo))/length(test[,response])
print(perc_observations_inside_interval)

# If you have used the default settings you will find that the proportion of observations within the interval  
# approximately matches the set confidence level of 95%. Thus, we were able to calculate an accurate prediction interval
# for new observations in a reasonable time frame.  
