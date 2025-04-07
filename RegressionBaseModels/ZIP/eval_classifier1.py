from LogisticRegression import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

irisDataset = datasets.load_iris()
X = irisDataset.data[:, [2, 3]]  # Petal length and width
y = (irisDataset.target != 0) * 1

scaler = StandardScaler()
X = scaler.fit_transform(X)

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.1, shuffle=True)

model = LogisticRegression(learning_rate=0.01, max_iter=1000)

model.weights = np.load('logistic_regression_weights.npy')
model.bias = np.load('logistic_regression_bias.npy')

print(f"Evaluating on test set of size: {len(yTest)}")
predictions = model.predict(XTest)
accuracy = np.mean(predictions == yTest)
print(f'Accuracy of Logistic Regression Model: {accuracy}')
