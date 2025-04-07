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
        import numpy as np

        self.batch_size = batch_size
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience

        self.weights = None
        self.bias = None
        self.training_loss = []

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
        # Reshaped y because... When calculating error, there was a mismatch between
        # y and error 
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience

        import numpy as np

        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, y.shape[1]))
        self.bias = np.zeros((1, y.shape[1]))

        valSplit = 0.1
        valSize = int(n_samples * valSplit)
        XTrain, XVal = X[:-valSize], X[-valSize:]
        yTrain, yVal = y[:-valSize], y[-valSize:]

        bestWeights = self.weights
        bestBias = self.bias
        bestValLoss = float('inf')
        patienceCounter = 0

        for epoch in range(self.max_epochs):
            indices = np.arange(XTrain.shape[0])
            # np.random.shuffle(indices)
            XTrain, yTrain = XTrain[indices], yTrain[indices]

            # Mini-batch gradient descent
            for start in range(0, XTrain.shape[0], self.batch_size):
                end = start + self.batch_size
                XBatch, yBatch = XTrain[start:end], yTrain[start:end]

                # Compute predictions
                yPred = np.dot(XBatch, self.weights) + self.bias

                # Compute gradients
                error = yPred - yBatch
                gradientWeights = (2 / XBatch.shape[0]) * np.dot(XBatch.T, error) + 2 * self.regularization * self.weights
                gradientBias = (2 / XBatch.shape[0]) * np.sum(error, axis=0)

                # RIP - here lies learning rate
                # weights = weights - learning_rate * gradientWeights
                # bias = bias - (learning_rate * gradientBias )
                # Update weights and bias

                self.weights -= 0.01 * gradientWeights
                self.bias -= 0.01 * gradientBias

            # Validate the model
            valPred = np.dot(XVal, self.weights) + self.bias
            valLoss = np.mean((valPred - yVal) ** 2)

            # Record training loss
            self.training_loss.append(valLoss)

            # Early stopping
            if valLoss < bestValLoss:
                bestValLoss = valLoss
                bestWeights = self.weights
                bestBias = self.bias
                patienceCounter = 0
            else:
                patienceCounter += 1
                if patienceCounter >= self.patience:
                    break

        # Set the best weights and bias
        self.weights = bestWeights
        self.bias = bestBias

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        import numpy as np
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
        import numpy as np
        yPred = self.predict(X)
        return np.mean((y - yPred) ** 2)
    def save(self, filePath):
        """Save the model parameters to a file."""
        import pickle
        with open(filePath, 'wb') as f:
            pickle.dump({'weights': self.weights, 'bias': self.bias}, f)

    def load(self, filePath):
        """Load the model parameters from a file."""
        import pickle
        with open(filePath, 'rb') as f:
            params = pickle.load(f)
            self.weights = params['weights']
            self.bias = params['bias']
