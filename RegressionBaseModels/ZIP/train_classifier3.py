from LogisticRegression import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

irisDataset = datasets.load_iris()
X = irisDataset.data
y = (irisDataset.target != 0) * 1

scaler = StandardScaler()
X = scaler.fit_transform(X)

unique, counts = np.unique(y, return_counts=True)
print(f"Class distribution in dataset: {dict(zip(unique, counts))}")

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.1)

model = LogisticRegression(learning_rate=0.01, max_iter=1000)

model.fit(XTrain, yTrain)

# yes I tried to plot this before realizing what I was doing.
# plot_decision_regions(XTest, yTest, clf=model, legend=2)
# plt.title('Decision Regions for All Features')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.savefig('train_classifier3.png', dpi=300, bbox_inches='tight')
# plt.close()
# plt.savefig('train_classifier3.png', dpi=300, bbox_inches='tight')
# plt.close()

predictions = model.predict(XTest)
print(f"Predictions: {predictions}")
print(f"Actual: {yTest}")
np.save('logistic_regression_weights_3.npy', model.weights)
np.save('logistic_regression_bias_3.npy', model.bias)
