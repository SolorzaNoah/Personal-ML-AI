from LinearRegression import LinearRegression
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

irisDataset = datasets.load_iris()
X = irisDataset.data
y = irisDataset.target

XTrain, XTest, yTrain, yTest = train_test_split(X[:, [2, 1]], X[:, 3], test_size=0.1)

model = LinearRegression()

model.load('model_regression1.pkl')

mse = model.score(XTest, yTest)
print(f'Mean Squared Error for Regression Model 1: {mse}')
