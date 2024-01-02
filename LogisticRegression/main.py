from model import LogisticRegression

import numpy as np

X = np.array([[10, 0, -10], [8.3, 1, -8.3], [9, 0.5, -9], [8.8, 1.1, -8.8], [7, 2, -7], [1, 10, -1], [0, 9.8, 0], [1.1, 7, -1.1], [0.5, 8, -0.5], [0.1, 10, -0.1]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

n_features = X.shape[1]

linear_model = LogisticRegression(
    n_features=n_features,
    learning_rate=0.05)

linear_model.train(X, y, 60)
