from LogisticRegression import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

iris_dataset = datasets.load_iris()
X = iris_dataset.data
y = (iris_dataset.target != 0) * 1  # binary shminary

# Use sepal length and width
X = X[:, [0, 1]]

scaler = StandardScaler()
X = scaler.fit_transform(X)

unique, counts = np.unique(y, return_counts=True)
print(f"Class distribution in dataset: {dict(zip(unique, counts))}")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print(f"Training set size: {len(y_train)}, Test set size: {len(y_test)}")

# print(f"Training data (X): {X_train}")
# print(f"Training targets (y): {y_train}")
model = LogisticRegression(learning_rate=0.01, max_iter=1000)

model.fit(X_train, y_train)

plot_decision_regions(X_test, y_test, clf=model, legend=2)
plt.title('Decision Regions for Sepal Features')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.savefig('train_classifier2.png', dpi=300, bbox_inches='tight')
plt.close()

predictions = model.predict(X_test)
print(f"Predictions: {predictions}")
print(f"Actual: {y_test}")
np.save('logistic_regression_weights_2.npy', model.weights)
np.save('logistic_regression_bias_2.npy', model.bias)
