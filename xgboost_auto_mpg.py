#
# Author: Jamey Johnston, Craig Mann
# Title: SciKit Learn Example with pickle
# Date: 2020/01/17
# Email: jameyj@tamu.edu, craigmann@tamu.edu
# Texas A&M University - MS in Analytics - Mays Business School
#

# Train models for 
# Save model to file using pickle
# Load model and make predictions
# Data From Kaggle
# https://www.kaggle.com/uciml/autompg-dataset/version/3

# Import OS and set CWD
import os
from settings import APP_ROOT

import numpy as np
from numpy import loadtxt, vstack, column_stack
import xgboost
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import pickle to save ML models
import pickle

# Load the Auto MPG Dataset
dataset = pd.read_csv(os.path.join(APP_ROOT, "Assignment-1---Craig-Mann\\auto-mpg.csv"), header=0, na_values="?", comment="\t")

# Headers of Data
# "mpg","cylinders","displacement","horsepower","weight","acceleration","model year","origin","car name"

# Drop car names because we don't care about this for this example because of added dimensionality
del dataset['car name']


# Check to see if there are any unknown values in the input.
dataset.isna().sum()

# There are 6 horsepower values missing, let's impute with the average.
dataset = dataset.fillna(dataset.mean())
# Checking again shows no more na values.
dataset.isna().sum()

# The origin column is actually categorical not ordinal or continuous.
dataset['origin'] = dataset['origin'].map(lambda x: {1: 'USA', 2: 'Europe', 3: 'Japan'}.get(x))

dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
dataset.tail()

# Make X cols by removing mpg.
cols = list(dataset)
cols = cols[1:]

# Describe the data to see some statistics
dataset.describe()

# Split the mpg data into X (independent variable) and y (dependent variable)
X = dataset[cols].astype(float)
Y = dataset['mpg'].astype(int)

# Split auto mpg data into train and validation sets
seed = 7
test_size = 0.3
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

# Fit model on Auto MPG Data using eXtendeded Gradient Boosting Regressor
modelXGB = xgboost.XGBRegressor()
modelXGB.fit(X_train, y_train)

# Make predictions for Validation data
y_predXGB = modelXGB.predict(X_valid)
predictionsXGB = [round(value) for value in y_predXGB]

# Evaluate predictions
mseXGB = mean_squared_error(y_valid, predictionsXGB)
rmseXGB = np.sqrt(mseXGB)
maeXGB = mean_absolute_error(y_valid,predictionsXGB)
print(f"Mean Square Error - XGB: {mseXGB}")
print(f"Root Mean Square Error - XGB: {rmseXGB}")
print(f"Mean Absolute Error - XGB: {maeXGB}")

# Create Dataset with Prediction and Inputs
predictionResultXGB = column_stack(([X_valid, vstack(y_valid), vstack(y_predXGB)]))

# Fit model on Wine Training Data using Random Forest save model to Pickle file
modelRF = RandomForestRegressor()
modelRF.fit(X_train, y_train)

# Make predictions for Validation data
y_predRF = modelRF.predict(X_valid)
predictionsRF = [round(value) for value in y_predRF]

# Evaluate predictions
mseRF = mean_squared_error(y_valid, predictionsRF)
rmseRF = np.sqrt(mseRF)
maeRF = mean_absolute_error(y_valid,predictionsRF)
print(f"Mean Square Error - RF: {mseRF}")
print(f"Root Mean Square Error - RF: {rmseRF}")
print(f"Mean Absolute Error - RF: {maeRF}")

# Still have over 10% higher than the actual mean mpg - 23.5 so maybe not that great of a model.

# Create Dataset with Prediction and Inputs
predictionResultRF = column_stack(([X_valid, vstack(y_valid), vstack(y_predRF)]))

# save model to file
pickle.dump(modelRF, open("autompg.pickleRF.dat", "wb"))

# Load model from Pickle file
loaded_modelRF = pickle.load(open("autompg.pickleRF.dat", "rb"))

# Predict MPG from inputs
# actual for this is 27.9...
loaded_modelRF.predict([[ 4,156,105,2800,14.4,80,1,0,0]])
# seems to do pretty great in predicting, actually... let me know what you think!

