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

        return cov

    def variance(
        self,
        X: list
    ) -> float:

        var = 0
        sampleMean = self.mean(X)
        for x in X:
            var += (x - sampleMean)**2

        return var

    def train(
        self,
        X: list,
        Y: list,
    ):

        xMean = self.mean(X)
        yMean = self.mean(Y)

        self.W[1] = self.covariance(X,Y)/self.variance(X)
        self.W[0] = yMean - self.W[1]*xMean

    def predict(
        self,
        x: float
    ):
        return self.W[0] + self.W[1]*x

    def render_init(
        self,
        X: list,
        Y: list,
    ):

        # Set lim
        x_min = np.min(X) - 1
        x_max = np.max(X) + 1
        line_range = np.arange(x_min, x_max, 0.1)

        # Plot data
        self.ax.scatter(X, Y,
                        s=80,
                        color='blue',
                        marker='o')

    def draw_line(
        self,
        X: list,
        Y: list,
        fn: str,
        line_color: str = "black",
    ):

        # Set lim
        x_min = np.min(X) - 1
        x_max = np.max(X) + 1
        line_range = np.arange(x_min, x_max, 0.1)

        line_x = [x for x in line_range]
        line_y = [self.W[0] + x*self.W[1] for x in line_range]

        self.ax.plot(line_x, line_y, color=line_color, label=fn)

        plt.savefig(f"./results/{fn}.png")

        self.ax.legend()


if __name__ == "__main__":
    X = [31.92, 33.55, 34.23, 34.97, 35.43, 36.38, 38.27]
    Y = [16.7, 15.9, 13.8, 14.8, 13.2, 12.2, 9.2]

    linear_model = LSM()
    linear_model.render_init(X, Y)

    linear_model.draw_line(X, Y, "before_train", line_color="red")

    linear_model.train(X, Y)

    linear_model.draw_line(X, Y, "after_train")

    plt.show()
