from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


# Least Square Method
class LSM:

    def __init__(
        self,
        is_render: bool = True
    ):

        # Randomly init the weights
        self.W = np.random.rand(2)

        self.__is_render = is_render

        # Rendering
        if self.__is_render:
            self.fig = plt.figure()
            plt.ion()  # Use interactive mode
            self.ax = self.fig.add_subplot(111)
            self.line = None


    @property
    def is_render(self) -> bool:
        return self.__is_render

    @is_render.setter
    def is_render(
        self,
        is_render: bool
    ):
        self.__is_render = is_render

    @staticmethod
    def mean(
        data: list
    ) -> float:
        return sum(data) / len(data)

    def covariance(
        self,
        X: list,
        Y: list,
    ) -> float:
        
        cov = 0
        xMean = self.mean(X)
        yMean = self.mean(Y)

        for x, y in zip(X, Y):
            cov += (x - xMean)*(y - yMean)

        cov = cov / len(X)

        return cov

    def variance(
        self,
        X: list
    ) -> float:

        var = 0
        sampleMean = self.mean(X)
        for x in X:
            var = (x - sampleMean)**2
        var = var / len(X)

        return var

    def train(
        self,
        X: list,
        Y: list,
    ):

        xMean = self.mean(X)
        yMean = self.mean(Y)

        print(self.covariance(X, Y))

        self.W[1] = self.covariance(X,Y)/self.variance(X)
        self.W[0] = yMean - self.W[1]*xMean

    def predict(
        self,
        x: float
    ):
        return self.W[0] + self.W[1]*x

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
        
        self.line, = self.ax.plot(x, y, color="black")

        plt.savefig(f"./results/{self.step}.png")
        plt.show()


if __name__ == "__main__":
    X = [31.92, 33.55, 34.23, 34.97, 35.43, 36.38, 38.27]
    Y = [16.7, 15.9, 13.8, 14.8, 13.2, 12.2, 9.2]

    linear_model = LSM()
    linear_model.train(X, Y)

    print(linear_model.W)
