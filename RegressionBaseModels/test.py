from LinearRegression import LinearRegression
from sklearn import datasets
import numpy as np

iris_dataset = datasets.load_iris()
x = iris_dataset.data
y = iris_dataset.target

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model
model.fit(x, y)

# Predict using the model
predictions = model.predict(x)

# Calculate the mean squared error
mse = model.score(x, y)

print("Predictions:", predictions)
print("Mean Squared Error:", mse)
