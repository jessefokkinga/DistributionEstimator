# Distribution estimator

Most data scientists will agree that is useful to quantify uncertainty when building machine learning models for regression problems. 
The measure of uncertainty is often provided by prediction intervals around the point predictions from the used model. 
This repository contains code to produce **scalable univariate prediction intervals** by estimating the full distribution of the target variable that is being modelled.
The method described, the *distribution estimator*, in this repository is a computionally efficient way to generate accurate prediction interval for any machine learning model of choice. 
It contain both an explanation of the idea, some details on the working of algorithm and two examples of implementations in Python and R. 

## The concept

Two commonly used ways to generate prediction intervals for regression problems are given by (i) bootstrapping and (ii) quantile regression. 
The downside of bootstrapping it computationally intesnive and may not be feasible for complex models with significant training time. 
Quantile regression is more tricky to implement, as it requires altering the loss function of the machine learning model, and hence it is not always a feasible option. 

The **distribution estimator** describes an alternative way to generate prediction interval for **any machine learnig model of choice**.
The described method is compatible with all regression problems and implementation is straightforward.
The distribution estimator depends on the assumption that the outcome of some regression problem is normally distributed around the predictions of some learning model.
Moreover, it also assumes that the standard deviation of this normal distribution is depend on the features that are used in the training process of this model.


## The algorithm

Assume we have a dataset with known realizations of some response variable (target or outcome) and various explanatory variables (features).
We want to use this data to train a model that predicts the response variable of new observations (for which the response variable is unknown) based on the same explanatory variables. 
In the method below, we refer to the data that we use to fit our models as the 'available data' and the observations for which we want to predict the response variable as the 'new observations'. 

1. Randomly split the available data in a train and a validation set
2. Fit a model based on observations in the train set, use this model to compute a prediction for each observation in the validation set 
3. Compute the squared residuals for each observation in the validation set. Save these residuals and the indices of the observations corresponding to these residuals
4. Repeat steps 1-3 multiple times
5. Fit a model that predicts the value of target based on the features (we use the full available data for this)
6. Fit a model that predicts the squared residual of any observations based on the features in the data. This model is based on the data that we saved in each replication of step 3. 
7. Compute the predicted value of y for each new observation based on the model of step 5
8. Compute the predicted squared residual for each new observations based on the features in the data and the model that we obtained during step 6
9. Compute the square root of the predicted squared residuals to obtain an approximated standard error for each new observation
10. The 95% prediction interval (estimated distribution) of any new observation is now equal to the predicted value of that observation plus and minus 1.96 times the approximated standard devation (note that the value 1.96 refers to the approximate t-statistic of a normal distribution related to a p-value of 0.05).

## Deployment

In order to deploy this idea to a problem of choice, this repository provides two templates that implement the described idea to a specific regression problem. Both implementations use a random forest model, but any algorithm suited for regression problems can be used as well. 
The implementation code contains extensive comments in order to explain each of the steps needed to build prediction intervals with the *distribution estimator*. 

