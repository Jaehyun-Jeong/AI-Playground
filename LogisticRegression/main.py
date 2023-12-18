from model import LinearModel
import matplotlib.pyplot as plt

import numpy as np

X = np.array([[10, 0], [8.3, 1], [9, 0.5], [8.8, 1.1], [7, 2], [1, 10], [0, 9.8], [1.1, 7], [0.5, 8], [0.1, 10]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

linear_model = LinearModel(learning_rate=0.05)
linear_model.train(X, y, 60)
