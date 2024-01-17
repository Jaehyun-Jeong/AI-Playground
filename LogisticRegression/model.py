import sys
sys.path.append("../")  # to import module

'''
에러 핸들링을 위한 함수
"denominator": 0으로 나누는 경우를 처리하기 위한 함수
"log_prob": 로그에 0이 들어간 경우를 처리하기 위한 함수
'''
from utils import denominator, log_prob

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


# Logistic Regression의 클래스
class LogisticRegression:

    def __init__(
        self,
        n_features: int = 2,
        is_train: bool = True,
        learning_rate: float = 0.1,
        is_render: bool = True
    ):

        # 피처의 수를 저장한다.
        self._n_features = n_features

        # 가중치를 랜덤하게 부여한다.
        self.W = np.random.rand(n_features + 1)

        # 학습중인 그래디언트를 저장한다.
        self._model_grad = 0
        self._sigmoid_grad = 0
        self._binary_cross_entropy_grad = 0

        # 학습중인지를 결정하는 인스턴스 변수이다.
        self.is_train = is_train

        # Learning rate 를 저장하는 인스턴스 변수이다.
        self.learning_rate = learning_rate

        # Feature 가 2개인 경우만 렌더링이 가능하다.
        if n_features == 2:
            self.is_render = is_render
        else:
            self.is_render = False

        # 진행한 학습 스텝 수를 저장한다.
        self.step = 0

        # 렌더링을 위한 초기화를 한다.
        if self.is_render:
            self.fig = plt.figure()
            # 그래프가 움직일 수 있도록
            # interactive mode 를 실행한다.
            plt.ion()  # interactive mode
            self.ax = self.fig.add_subplot(111)
            self.line = None

    # 모든 열에 대해, 선형함수를 거치는 메소드이다.
    # 계산 방법은 AI PLAYGROUND의 "모델, 최적화 가이드라인"을 참고
    def model(
        self,
        data: np.ndarray
    ) -> np.ndarray:

        # 데이터의 크기를 구한다.
        data_size = data.shape[0]

        # 데이터에 바이어스 항을 위한 열을 추가한다.
        ones = np.ones(data_size)
        X = np.column_stack((ones, data))

        # 학습 중이라면, 그래디언트를 계산해서 저장한다.
        if self.is_train:
            # (|N|, features)
            self._model_grad = X

        # (|N|, )
        return np.matmul(X, self.W)

    # 시그모이드 함수의 메소드이다.
    def sigmoid(
        self,
        arr: np.ndarray
    ) -> np.ndarray:

        # 학습 중이라면, 그래디언트를 계산해서 저장한다.
        if self.is_train:
            # (|N|, )
            self._sigmoid_grad = np.exp(-arr) / (1 + np.exp(-arr))**2

        # (|N|, )
        return 1 / (1 + np.exp(-arr))

    # Binary cross entropy 함수의 메소드이다.
    def binary_cross_entropy(
        self,
        Y_hat: np.ndarray,  # 예측 값
        Y: np.ndarray  # 실제 값
    ) -> np.ndarray:

        # 그래디언트를 계산해서 저장한다.
        if self.is_train:
            # (|N|, )
            self._binary_cross_entropy_grad = \
                (Y_hat - Y) / denominator(Y_hat*(1 - Y_hat))

        # (|N|, )
        return -(Y*log_prob(Y_hat) + (1-Y)*log_prob(1-Y_hat))

    # Batch gradient descent 를 사용해서 최적화를 진행한다.
    def batch_gradient_descent(
        self
    ):

        # 그래디언트를 저장한다.
        # "self._n_features + 1"에서 1을 더하는 이유는, bias 항이 있기 때문이다.
        temp_grad = np.zeros(self._n_features + 1)
        grad = np.zeros(self._n_features + 1)

        # Batch Gradient Descent 에서 batch 의 크기는 전체 데이터의 크기이다.
        batch_size = self._model_grad.shape[0]

        
        for idx, model_grad in enumerate(self._model_grad):
            temp_grad = np.copy(model_grad)
            temp_grad = self._sigmoid_grad[idx] * temp_grad
            temp_grad = self._binary_cross_entropy_grad[idx] * temp_grad
            grad = grad + temp_grad

        # Get average of the gradients
        grad = grad / batch_size

        # Update the weights
        self.W = self.W - self.learning_rate * grad

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

        if self.is_render:
            self.render_init(X, Y)

        for epoch in range(epochs):
            L = self.forward(X, Y)
            self.batch_gradient_descent()
            self.step += 1

            if self.is_render:
                self.render_update(X)

            # print training status
            resultStr = f"epoch:{epoch} loss: {round(L, 6)}"
            resultLen = len(resultStr)
            print(resultStr)
            print('-'*resultLen)

if __name__ == "__main__":
    # data = np.array([[1, 2], [3, 2], [10, 2.3]])
    X = np.array([[1, 0], [3, 2], [10, 2.3]])
    Y = np.array([0, 0, 1])

    linear_model = LogisticRegression()
    linear_model.train(X, Y, 200)
