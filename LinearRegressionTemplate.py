#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

class LinearRegression:
    def __init__(self):
        self.theta = []
        
    def fit(self,X_train,Y_train):
        X = np.append(arr = np.ones((X_train.shape[0],1)).astype(int), values = X_train, axis = 1)
        
        from numpy.linalg import inv #to inverse matrix
        self.theta = np.matmul(np.matmul(inv(np.matmul(X.transpose(), X)),X.transpose()),Y_train)
        self.theta = np.reshape(self.theta,(self.theta.shape[0],1))
        
    def predict(self,X_test):
        X = np.append(arr = np.ones((X_test.shape[0],1)).astype(int), values = X_test, axis = 1)
        Y_pred = np.matmul(self.theta.transpose(),X.transpose())
        Y_pred = Y_pred.transpose()
        return Y_pred
    
    def coefficient(self):
        return self.theta

#Modify code below to preprocess data according to your data
#Data Preprocessing   
#importing dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
#dummy encoding (making seprate columns for seprate categories )
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
#Avoiding dummy variale trap
X = X[:, 1:]

#Splitting the dataset into training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Applying LinearRegression
regressor = LinearRegression() #making object of LinearRegression class
regressor.fit(X_train,Y_train) #Fitting reggressor to X_train and Y_train
y_pred = regressor.predict(X_test) #Predicting X_test
theta = regressor.coefficient() #It stores coefficients of independent vaiables derived from regression
