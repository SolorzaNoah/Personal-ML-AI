from LogisticRegression import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

irisDataset = datasets.load_iris()
X = irisDataset.data[:, [2, 3]]  # Petal length and width
y = (irisDataset.target != 0) * 1

scaler = StandardScaler()
X = scaler.fit_transform(X)

unique, counts = np.unique(y, return_counts=True)
# print(f"Class distribution in dataset: {dict(zip(unique, counts))}")

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.1)

model = LogisticRegression(learning_rate=0.01, max_iter=1000)

model.fit(XTrain, yTrain)

plot_decision_regions(XTest, yTest, clf=model, legend=2)
plt.title('Decision Regions for First Two Features')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

plt.savefig('train_classifier1.png', dpi=300, bbox_inches='tight')

plt.close()

predictions = model.predict(XTest)
print(f"Predictions: {predictions}")
print(f"Actual: {yTest}")
np.save('logistic_regression_weights.npy', model.weights)
np.save('logistic_regression_bias.npy', model.bias)
