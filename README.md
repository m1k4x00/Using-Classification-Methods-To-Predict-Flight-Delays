# Using-Classification-Methods-To-Predict-Flight-Delays

## 1. Introduction

The airline industry is a massive operation with countless intricate moving parts. It is a fact that
sometimes flights are delayed affecting the lives of many people. The goal of this project is to see if
it is possible to predict whether a flight is delayed based on the information provided prior to the
flight taking place.
Problem formulation and the used dataset will be discussed in the second section after which we
will go on to discuss the feature selection and modeling in the third section. Hypothesis space, loss
function and model validation will be discussed in section 3.2. In section 4 we are going to go over
the results which will be followed by the conclusion of the report in section 5. The code used in this
project can be found as an appendix to this report and references in section 6.

## 2. Problem Formulation

The dataset used in this project contains data from 600 000 flights mainly in the United States. In
this case a single flight is used as a data point. Each data point contains the airline, type of aircraft,
source and destination airports, day of the week, time of the day, and the length of the flight.
Further explanation of the data and its type is provided below in table 1. The dataset was obtained
from Kaggle[1] and was originally provided for a data expo competition as a challenge task. There
are no null values, and all data points are unique.
The objective is to predict whether the flight is delayed or not. Therefore, we choose delay as our
label and classify the input given to the model to be either 0 or 1 which corresponds to not delayed
and delayed, respectively. The dataset contains both strings and integers. For this model we are
going to pick which features are the most favorable for the prediction, thus making the model
supervised. Feature selection is discussed further in the next section.

![image](https://github.com/m1k4x00/Using-Classification-Methods-To-Predict-Flight-Delays/assets/142576207/3576d9c3-48cd-4c60-b679-b234127f4bdd)

## 3. Methods
### 3.1 Feature Selection

The features were selected by considering their usefulness in the given problem. For example, the
type of aircraft most likely won’t make the flight late but the time of the day might. Departure and
arrival airports might be important contributors to making a flight late but when analyzing the data,
we can see that because there are so many different combinations of departure and arrival airports,
it would not be beneficial to our model.
When inspecting the different airlines shown in Figure 1, we can see that selecting the airline
column seems like a valid choice for a feature. Different airlines have different amounts of delays in
relation to the flights being on time which can help our model make better predictions. Because the
airline data is categorical, and our model needs numerical data, we will need to convert the data into
numbers. This can be done by creating dummy variables from the airline names.
When looking at the correlation matrix of our data shown in table 2, we can see that the features
that contain integer values are mostly weakly correlated to the label. This means that they are only
somewhat useful for the model, but we are going to use them anyway because our data is otherwise
too limited. One explanation for the low correlation is the fact that there are countless variables
affecting the flights for example weather which are simply not predictable.

![image](https://github.com/m1k4x00/Using-Classification-Methods-To-Predict-Flight-Delays/assets/142576207/cb82bacd-2bb1-4c0d-b8a1-49baab6ae3f9)
![image](https://github.com/m1k4x00/Using-Classification-Methods-To-Predict-Flight-Delays/assets/142576207/1ca07833-010b-4e0a-9624-fa18629986c9)

### 3.2 Modeling
### 3.2.1 Logistic Regression

When inspecting the data, it seems that a classification method would best fit our purposes. We can
see that the delay column is already in a binary shape, and we have several features which will be
used to predict the label value. Given that our dataset is quite large, the model that we use should be
as efficient as possible to shorten the training time. Thus, we will use logistic regression as the first
model for it is quite efficient to train. Even though, the correlation between the features and the
label is not strong, logical regression is still a good starting point when trying to classify data. In
this case the model accuracy will most likely not be remarkably high given the nature of the data
and the problem.
The loss function best suited for our purposes at this stage of our project is logistic loss as premade
libraries already exist for logistic regression which uses logistic loss as default. In this case we are
going to use the Scikit-learn library. Logistic loss is also the most used loss function in logistic
regression.

### 3.2.2 Multilayer Perceptron (MLP)

For our second method we are going to use an MLP model. By using a deep a learning model, we
can hope to add complexity to our model and thus improve prediction accuracy. Because MLP uses
backpropagation, it might classify data which is not linearly separable better. Though, it is much
slower to train than for example logistic regression. However, it is unlikely that using a different
model will improve the prediction accuracy of the model drastically given the nature of our data.
MLP consists of an input layer, hidden layers, and a binary output layer which we have chosen
arbitrarily by testing different values. As the activation function rectified linear unit (ReLU) was
chosen for it is often used with deep learning models and it is easy to train. Neural networks often
use a similar loss function as logistic loss which is used in logistic regression. We are going to use
logistic loss as it is already implemented in the Scikit-learn library and because it is often used with
MLP models.

### 3.3 Model Validation

For both models we are going to use a train-test split to create the training, validation, and test sets.
Because we have a large dataset, we can fit 80% of our data into the training set and 20% of the
data into the validation and test sets. As of the validation and test sets, we simply split the data into
two 10% sets. Therefore, the data will be split into an 80% training set, a 10% validation set and a
10% test set. By using a validation set we can further reduce the risk of over-fitting the data if we
want to tune any hyperparameters while training the model. We are not using k-fold cross validation

## 4. Results

Because both logistic regression and multilayer perceptron uses logistic loss as their loss functions,
it seems natural to also use logistic loss to calculate the training and validation errors to compare the
models more easily. The calculated errors and accuracies for both methods can be observed in table
2. We can see that the validation accuracies are both similar, but the validation error of MLP is
visibly lower. It seems that neither of the models are overfitted based on the training and validation
errors. Based on this information, the better model would be MLP, though, with only a small
margin. Thus, we choose MLP as our final method. The testing error and accuracy of the MLP
model can be seen in table 2. The testing set was created in section 3.3 by splitting the data into
training, validation, and test sets to allow hyperparameter tuning which we did for the MLP model.
because we have a large dataset, and thus it would be computationally too costly. [2]

![image](https://github.com/m1k4x00/Using-Classification-Methods-To-Predict-Flight-Delays/assets/142576207/9cc7de2d-55f9-4f5a-aaa4-22cbe8781521)

## 5. Conclusion

The goal of this report was to see if it is possible to predict whether a flight is delayed before the
flight has taken place using logistic regression and multilayer perceptron models. Based on our
model we can predict whether a flight is delayed with approximately 65% percent accuracy. Given
the nature of the problem and the limitations of our data the result is acceptable and expected.
Though, generally we would want our model to be in the accuracy range of 70-90%. Because a
delayed flight can be the result of many different variables, most of the time a flight can be delayed
by unpredictable reasons. For example, bad weather or a malfunction might have been the cause of
the delay which is not possible to predict using our data.
It seems that our model wasn’t overfitted and worked correctly given untested data, but it can be
certainly improved. For example, we could formulate the problem stricter and perhaps use more
meaningful features. Now our range of different features are quite broad with features which
doesn’t correlate strongly with the label. Also getting more meaningful data would certainly
improve the model e.g., the current dataset doesn’t give us the date of the flight, so we don’t have
any data on seasonality of the delays.

## 6. References

[1] Airline Dataset to predict a delay
https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-adelay?resource=download
[2] Hyperparameters for Classification Machine Learning Algorithms
https://machinelearningmastery.com/hyperparameters-for-classification-machine-learningalgorithms/
[3] Scikit-learn machine learning library for python
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
https://scikitlearn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html?highlight=mlp#skl
earn.neural_network.MLPClassifier
