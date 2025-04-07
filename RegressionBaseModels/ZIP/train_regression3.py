from LinearRegression import LinearRegression
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

irisDataset = datasets.load_iris()
X = irisDataset.data
y = irisDataset.target

XTrain, XTest, yTrain, yTest = train_test_split(X[:, [0]], X[:, 1], test_size=0.1)

# print(f"Training data (X): {XTrain}")
# print(f"Training targets (y): {yTrain}")
model = LinearRegression()

model.fit(XTrain, yTrain, batch_size=32, regularization=0.1, max_epochs=100, patience=3)
model.save('model_regression3.pkl')

plt.plot(model.training_loss)
plt.title('Training Loss for Regression Model 3')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig('training_loss_regression3.png')
plt.close()
