from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


class LinearModel:

    def __init__(
        self,
        is_train: bool = True,
        learning_rate: float = 0.1,
        is_render: bool = True
    ):

        # Rnadomly init the weights
        self.W = np.random.rand(3)

        # Gradients
        self.__model_grad = 0
        self.__sigmoid_grad = 0
        self.__binary_cross_entropy_grad = 0

        # Train parameters
        self.__is_train = is_train
        self.__learning_rate = learning_rate
        self.__is_render = is_render

        # epoch step number
        self.step = 0

        # Rendering
        if self.__is_render:
            self.fig = plt.figure()
            plt.ion()  # Use interactive mode
            self.ax = self.fig.add_subplot(111)
            self.line = None

    @property
    def is_train(self) -> bool:
        return self.__is_train

    @is_train.setter
    def is_train(
        self,
        is_train: bool
    ) -> bool:
        self.__is_train = is_train

    @property
    def learning_rate(self) -> float:
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(
        self,
        learning_rate: float
    ) -> float:
        self.__learning_rate = learning_rate

    @property
    def is_render(self) -> bool:
        return self.__is_render

    @is_render.setter
    def is_render(
        self,
        is_render: bool
    ):
        self.__is_render = is_render

    def model(
        self,
        data: np.ndarray
    ) -> np.ndarray:

        # Get the number of data
        data_size = data.shape[0]
        ones = np.ones(data_size)  # For bias multiplication

        # Add column of ones to multiply bias
        X = np.column_stack((ones, data))

        # Save grads
        if self.__is_train:
            # (|N|, features)
            self.__model_grad = X

        # (|N|, )
        return np.matmul(X, self.W)

    def sigmoid(
        self,
        arr: np.ndarray
    ) -> np.ndarray:

        # Save grads
        if self.__is_train:
            # (|N|, )
            self.__sigmoid_grad = np.exp(-arr) / (1 + np.exp(-arr))**2

        # (|N|, )
        return 1 / (1 + np.exp(-arr))

    def binary_cross_entropy(
        self,
        Y_hat: np.ndarray,
        Y: np.ndarray
    ) -> np.ndarray:

        if self.__is_train:
            # (|N|, )
            self.__binary_cross_entropy_grad = \
                (Y_hat - Y) / (Y_hat*(1 - Y_hat))

        # (|N|, )
        return -(Y*np.log(Y_hat) + (1-Y)*np.log(1-Y_hat))

    def batch_gradient_descent(
        self
    ):

        temp_grad = np.zeros(3)
        grad = np.zeros(3)
        batch_size = self.__model_grad.shape[0]

        for idx, model_grad in enumerate(self.__model_grad):
            temp_grad = np.copy(model_grad)
            temp_grad = self.__sigmoid_grad[idx] * temp_grad
            temp_grad = self.__binary_cross_entropy_grad[idx] * temp_grad
            grad = grad + temp_grad

        # Get average of the gradients
        grad = grad / batch_size

        # Update the weights
        self.W = self.W - self.__learning_rate * grad

    def forward(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> np.float64:

        Y_hat = self.model(X)
        Y_hat = self.sigmoid(Y_hat)
        L = self.binary_cross_entropy(Y_hat, Y)

        return np.mean(L)

    # If f = 0, sigmoid(f) = 0.5.
    # So, decision boundary is equation f = 0
    def decision_boundary(
        self,
        X: np.ndarray
    ) -> Tuple[list, list]:

        # Get min, max value from data
        min_value = np.min(X[:, 0]) - 1
        max_value = np.max(X[:, 0]) + 1
        line_range = np.arange(min_value, max_value, 0.1)

        x = [x for x in line_range]
        y = [-((self.W[0] + self.W[1]*x) / self.W[2]) \
            for x in line_range]

        return x, y

    def render_init(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ):
        # Set lim
        x_min = np.min(X[:, 0]) - 1
        x_max = np.max(X[:, 0]) + 1
        y_min = np.min(X[:, 1]) - 1
        y_max = np.max(X[:, 1]) + 1
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)

        # Plot data
        self.ax.scatter(X[Y == 0, 0], X[Y == 0, 1],
                        s=80,
                        color='red',
                        marker='*')
        self.ax.scatter(X[Y == 1, 0], X[Y == 1, 1],
                        s=80,
                        color='blue',
                        marker='D')
        
        x, y = self.decision_boundary(X)
        self.line, = self.ax.plot(x, y, color="black")

        plt.savefig(f"./results/{self.step}.png")
        plt.show()

    def render_update(
        self,
        X: np.ndarray,
    ):
        x, y = self.decision_boundary(X)

        # Update
        self.line.set_ydata(y)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        plt.savefig(f"./results/{self.step}.png")

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 10,
    ):
        if self.__is_render:
            self.render_init(X, Y)

        for _ in range(epochs):
            L = self.forward(X, Y)
            self.batch_gradient_descent()
            self.step += 1

            if self.__is_render:
                self.render_update(X)
            print(L)

if __name__ == "__main__":
    # data = np.array([[1, 2], [3, 2], [10, 2.3]])
    X = np.array([[1, 0], [3, 2], [10, 2.3]])
    Y = np.array([0, 0, 1])

    linear_model = LinearModel()
    linear_model.train(X, Y, 60)
