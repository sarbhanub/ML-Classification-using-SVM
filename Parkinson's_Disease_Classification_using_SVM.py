# imports

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

# Loading the data from the directory

# data > pandas object
parkinsons_data = pd.read_csv('/content/data/parkinsons.csv')

# Getting an early insight over the data

parkinsons_data.head() # first 5 rows

# Checking for null values

parkinsons_data.info() # data check

# No. of Rows and Columns

parkinsons_data.shape # no of rows and cols

# Checking for missing values

parkinsons_data.isnull().sum() # check for missing values

# Checking Data Status Ratio for our Classification Problem
# status variable percentage 
# 1 = Positive | 0 = Negative
parkinsons_data['status'].value_counts()

parkinsons_data.groupby('status').mean() # insights about distinctions for seperate variables and taking the mean

# Data Preprocessing


X = parkinsons_data.drop(columns=['name','status'], axis = 1) # dropping the column
Y = parkinsons_data['status']
print("\nX Variable Status\n")
print(X)
print("\nY Variable Status\n")
print(Y)

# Data Splitting | Training Data & Test Data

# taking 80% data for Training Data, 20% data for Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape) # check

print(Y.shape, Y_train.shape, Y_test.shape) # check

# Standardizing the Data

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

print(X_train)

# Training the Model using SVM

svm_model = svm.SVC(kernel='linear')

# training the model with X_train
svm_model.fit(X_train, Y_train)

# Model Evaluation

# Train Data Accuracy

X_train_predict = svm_model.predict(X_train) # prediction
accuracy_train = accuracy_score(Y_train, X_train_predict)*100 # getting score in percentage
print(accuracy_train) # Training accuracy

# Test Data Accuracy

X_test_predict = svm_model.predict(X_test)
accuracy_test = accuracy_score(Y_test, X_test_predict)*100 # getting score in percentage
print(accuracy_test)

# Saving the Model with Pickle
# using pickle to store model for saving computational resources

parkinsons_classifier_model = "/content/data/parkinsons_classifier_model.pickle"  
with open(parkinsons_classifier_model, 'wb') as file:  
    pickle.dump(svm_model, file)

# Outcomes: Data Prediction Examples

# Example: 1
# Known to us that Person is affected.
#imputing an arbitrary data from the set with known value of 1 for 'status' variable which is our outcome
input_data = [120.26700,137.24400,114.82000,0.00333,0.00003,0.00155,0.00202,0.00466,0.01608,0.14000,0.00779,0.00937,0.01351,0.02337,0.00607,24.88600,0.596040,0.764112,-5.634322,0.257682,1.854785,0.211756]
input_data_numpy = np.asanyarray(input_data)
input_data_reshape = input_data_numpy.reshape(1, -1) # predicting for one dat apoint value
standard_data = scaler.transform(input_data_reshape)
# Loading the Pickle model
parkinsons_classifier_model = pickle.load(open('/content/data/parkinsons_classifier_model.pickle','rb'))
score = parkinsons_classifier_model.score(X_test, Y_test)
print("Test score: {0:.2f} %".format(100 * score))
predict = parkinsons_classifier_model.predict(standard_data)
print(predict)

if (predict[0] == 0):
  print("Person is NOT affected.")

else:
  print("Person is affected.")

# Example: 2
# Known to us that Person isn't affected.
input_data = [197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569]
input_data_numpy = np.asanyarray(input_data)
input_data_reshape = input_data_numpy.reshape(1, -1)
standard_data = scaler.transform(input_data_reshape)

# parkinsons_classifier_model = pickle.load(open('/content/data/parkinsons_classifier_model.pickle','rb'))
# score = parkinsons_classifier_model.score(X_test, Y_test)
# print("Test score: {0:.2f} %".format(100 * score))

predict = parkinsons_classifier_model.predict(standard_data)
print(predict)

if (predict[0] == 0):
  print("Person is NOT affected.")

else:
  print("Person is affected.")

# Example: 3
# Known to us that Person **isn't affected**.

input_data = [128.00100,138.05200,122.08000,0.00436,0.00003,0.00137,0.00166,0.00411,0.02297,0.21000,0.01323,0.01072,0.01677,0.03969,0.00481,24.69200,0.459766,0.766204,-7.072419,0.220434,1.972297,0.119308]
input_data_numpy = np.asanyarray(input_data)
input_data_reshape = input_data_numpy.reshape(1, -1)
standard_data = scaler.transform(input_data_reshape)

# parkinsons_classifier_model = pickle.load(open('/content/data/parkinsons_classifier_model.pickle','rb'))
# score = parkinsons_classifier_model.score(X_test, Y_test)
# print("Test score: {0:.2f} %".format(100 * score))

predict = parkinsons_classifier_model.predict(standard_data)
print(predict)

if (predict[0] == 0):
  print("Person is NOT affected.")

else:
  print("Person is affected.")