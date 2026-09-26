import numpy as np


class LinearRegression:
    """
    Linear Regression implemented from scratch using NumPy
    and Batch Gradient Descent.

    Model:
        y_hat = Xw + b

    Cost function:
        J(w, b) = (1 / m) * sum((y - y_hat)^2)

    Gradient descent:
        w = w - learning_rate * dw
        b = b - learning_rate * db
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        """
        Initialize the linear regression model.

        Parameters
        ----------
        learning_rate : float
            Step size used during gradient descent.

        n_iterations : int
            Number of gradient descent iterations.
        """

        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")

        if n_iterations <= 0:
            raise ValueError("n_iterations must be greater than 0.")

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        # Model parameters
        self.weights = None
        self.bias = None

        # Training information
        self.cost_history = []

        # Dataset information
        self.n_training_examples = None
        self.n_features = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model using Batch Gradient Descent.

        Parameters
        ----------
        X : np.ndarray
            Training features with shape (m, n).

        y : np.ndarray
            Target values with shape (m,).

        Returns
        -------
        self
            Trained model.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        self._validate_training_data(X, y)

        # Number of training examples and features
        self.n_training_examples, self.n_features = X.shape

        # Initialize model parameters
        self.weights = np.zeros(self.n_features)
        self.bias = 0.0

        # Reset cost history in case fit() is called again
        self.cost_history = []

        # Gradient Descent
        for _ in range(self.n_iterations):

            # Make predictions
            y_prediction = self.predict(X)

            # Calculate gradients
            d_weights = (
                -2
                / self.n_training_examples
                * X.T.dot(y - y_prediction)
            )

            d_bias = (
                -2
                / self.n_training_examples
                * np.sum(y - y_prediction)
            )

            # Update parameters
            self.weights -= self.learning_rate * d_weights
            self.bias -= self.learning_rate * d_bias

            # Calculate and store cost
            cost = self._compute_cost(y, y_prediction)
            self.cost_history.append(cost)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the learned parameters.

        Parameters
        ----------
        X : np.ndarray
            Input features with shape (m, n).

        Returns
        -------
        np.ndarray
            Predicted target values.
        """

        if self.weights is None or self.bias is None:
            raise RuntimeError(
                "The model has not been trained yet. "
                "Call fit() before predict()."
            )

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2-dimensional array with shape (m, n)."
            )

        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, "
                f"but received {X.shape[1]}."
            )

        return X.dot(self.weights) + self.bias

    def _compute_cost(
        self,
        y: np.ndarray,
        y_prediction: np.ndarray
    ) -> float:
        """
        Calculate Mean Squared Error.

        J(w, b) = (1 / m) * sum((y - y_hat)^2)
        """

        errors = y - y_prediction

        return np.mean(errors ** 2)

    def _validate_training_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        """Validate training data before fitting the model."""

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2-dimensional array with shape (m, n)."
            )

        if y.ndim != 1:
            raise ValueError(
                "y must be a 1-dimensional array with shape (m,)."
            )

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        if X.shape[0] == 0:
            raise ValueError("Training data cannot be empty.")

        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values.")

        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or infinite values.")