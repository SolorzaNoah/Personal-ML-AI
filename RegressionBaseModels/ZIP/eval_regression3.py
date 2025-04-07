from LinearRegression import LinearRegression
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

irisDataset = datasets.load_iris()
X = irisDataset.data
y = irisDataset.target

XTrain, XTest, yTrain, yTest = train_test_split(X[:, [0]], X[:, 1], test_size=0.1)

model = LinearRegression()

model.load('model_regression3.pkl')

mse = model.score(XTest, yTest)
print(f'Mean Squared Error for Regression Model 3: {mse}')
