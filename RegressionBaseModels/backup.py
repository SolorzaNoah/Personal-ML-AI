import numpy as np

class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Fit a linear model.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience

        X = np.array(X)
        y = np.array(y)

        samples, features = X.shape
        self.weights = np.zeros((features, 1))
        self.bias = np.zeros((1, 1))

        # validationSplit = int(0.1 * samples)
        split = int(0.9 * samples)
        XTrain, XVal = X[:split], X[split:]
        yTrain, yVal = y[:split], y[split:]

        bestWeights = self.weights.copy()
        bestBias = self.bias.copy()
        # bestValLoss = 0
        bestValLoss = float("inf")
        epochsWithoutImprovement = 0

        for epoch in range(max_epochs):
            indices = np.arange(split)
            XTrain, yTrain = XTrain[indices], yTrain[indices]

            for i in range(0, len(XTrain), self.batch_size):
                XBatch = XTrain[i:i + self.batch_size]
                yBatch = yTrain[i:i + self.batch_size]
                
                # y = w0 + w1x1
                predictions = np.dot(XBatch, self.weights) + self.bias

                # error = how off our prediction was
                errors = predictions - yBatch


                print("Shape of XBATCH" ,np.shape(XBatch))
                print("\n\n\n\nShape of yBATCH" ,np.shape(yBatch))
                gradientWeights = (2 / len(XBatch)) * np.dot(XBatch.T, predictions - yBatch) + 2 * self.regularization * self.weights
                gradientBias = (2 / len(XBatch)) * np.sum(errors)

                # RIP - here lies learning rate
                # weights = weights - learning_rate * gradientWeights
                # bias = bias - (learning_rate * gradientBias )

                self.weights -= gradientWeights
                self.bias -= gradientBias

            valLoss = self.score(XVal, yVal)

            if valLoss < bestValLoss:
                bestValLoss = valLoss
                bestWeights = self.weights.copy()
                bestBias = self.bias
                epochsWithoutImprovement = 0
            else:
                epochsWithoutImprovement += 1
                if epochsWithoutImprovement >= self.patience:
                    break

        self.weights = bestWeights
        self.bias = bestBias

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        # TODO: Implement the prediction function.
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        # TODO: Implement the scoring function.
        yPred = self.predict(X)
        return np.mean((y - yPred) ** 2)
    
    def save(self, file_path):
        """Save the model parameters (weights and bias) to a file."""
        np.savez(file_path, weights=self.weights, bias=self.bias)
        print(f"Model parameters saved to {file_path}")


