import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy import stats

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

path              = "./data/winequality-white.csv"
data              = pd.read_csv(path, sep=";")
response          = "alcohol"
features          = data.columns[data.columns != response]

# Ultimately, we want to be able to predict the alcohol percentage for any new wine, based on a set of features and a some
# statistical model that is trained on a sample of wines for which the alcohol percentage is already known. In order to mimic 
# this objective, we split our data in a train and a test set. You can specify the size of your test set in the 
# 'size_test' parameter. In this template, we set this size equal to 10% of the full data set. 

n                  = len(data.index)
size_test          = int(round(n/10))
ind_test           = np.random.choice(range(1, n), size = size_test, replace = False)
train              = data.drop(data.index[ind_test])
test               = data.iloc[ind_test]


# Each prediction interval has an associated confidence level that roughly quantifies the desired probability 
# with which the response value of a new observation lies in the calculated prediction interval. In this template we use a 
#confidence level of 95%. 

confidence         = 0.95

# The parameter C defines the number of rounds of random subsampling validation that is used to calculate the residuals 
# for our error model. The parameter Q defines the size of the validation set that is used to validate the main model in 
# each subsampling validation replication.

# We also specify the two empty vectors in which we can save (i) the squared residuals and the (ii) indices corresponding 
# to each residual. 

Q                  = int(round(len(train.index)/10))
C                  = 20 
residuals          = []
indices            = []

# Now we will use the random subsampling validation to obtain the vector of squared residuals, so that we can quantify how far
# observations will approximately deviate from their predicted values. We will eventually use this vector of residuals 
# to esimate an observation-specific standard devation 

for i in range(C):
  cv     = np.random.choice(range(1,len(train.index)), size = Q, replace = False)
  fit_cv = RandomForestRegressor(n_estimators=100)
  fit_cv.fit(X = train[features].drop(train.index[cv]), y = train[response].drop(train.index[cv]))
  
  y_hat = fit_cv.predict(train[features].iloc[cv])
  e2_hat = (y_hat - train[response].iloc[cv])**2
  
  residuals = np.append(residuals, e2_hat)
  indices = np.append(indices, cv)


# We are now going to use all the squared residuals and the indices that we saved to build a model that can predict how
# far a prediction of our main statistical model deviates from its true value, based on the set of features in our data.
# We use a random forest model to predict the squared residuals, because this algorithm often achieves good accuracy and 
# requires little to no parameter tuning. However, in principle any other model could be used as well. 

sd_estimator = RandomForestRegressor(n_estimators=100)
sd_estimator.fit(train[features].iloc[indices], residuals)

# Now it is time to fit our main model. We fit a random forest model on our full training data. 
# Afterwards, we use our main model to predict the alcohol percentage of the observations in our test set. 

main_model = RandomForestRegressor(n_estimators=100)
main_model.fit(X = train[features], y =train[response])

pred = main_model.predict(test[features])

# We will now use our sd_estimator model to predict the deviation from the predicted value 
# for each observation in order to obtain an observation-specific approximated standard deviation.  

std = np.sqrt(sd_estimator.predict(test[features]))

# In order to calculate a prediction interval for each new observation, we use the approximated standard deviation for
# each observation in the test set and the assumption that the true realizations of our response variable (alcohol
# percentage) are normally distributed around the predicted values obtained from our main model. 

t_value = stats.t.ppf(1 - (1-confidence)/2, n - 1 )
up = pred + t_value*std
lo = pred - t_value*std

# Now we have obtained our desired result. That is, for each observation in our test set we have calculated a
# predicted value (pred), and an upper (up) and lower (lo) bound of the 95% prediction interval.
# In order to check the validity of our prediction interval, we will calculate what percentage of the observations
# in the test set fall in their calculated prediction interval. 

perc_observations_inside_interval = len(test.loc[(up>test[response]) & (test[response] > lo)].index)/len(test.index)
print(perc_observations_inside_interval)

# If you have used the default settings you will find that the proportion of observations within the interval  
# approximately matches the set confidence level of 95%. Thus, we were able to calculate an accurate prediction interval
# for new observations in a reasonable time frame.  
