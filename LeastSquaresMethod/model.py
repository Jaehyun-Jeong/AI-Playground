import numpy as np
import matplotlib.pyplot as plt


# Least Squares Methods 를 표현하는 클래스
class LSM:

    def __init__(
        self,
        is_render: bool = True  # 렌더링 여부를 결정한다.
    ):

        # 가중치를 랜덤하게 부여한다.
        self.W = np.random.rand(2)

        # 렌더링 여부를 결정하는 인스턴스 변수이다.
        self._is_render = is_render

        # 렌더링을 위한 초기화
        if self._is_render:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.line = None

    # 리스트의 평균을 구하는 메소드
    @staticmethod
    def mean(
        data: list
    ) -> float:
        # 다 더한 값을 "data"의 크기로 나눈다.
        return sum(data) / len(data)

    # "X", "Y"의 공분산을 구한다.
    def covariance(
        self,
        X: list,
        Y: list,
    ) -> float:

        cov = 0

        # "X", "Y"의 평균을 저장한다.
        xMean = self.mean(X)
        yMean = self.mean(Y)

        # 공분산의 계산과정이다.
        for x, y in zip(X, Y):
            cov += (x - xMean)*(y - yMean)

        # 원래 공분산은 "X"의 크기로 나누지만,
        # 전체적인 계산량을 고려해 나누는 식을 제거했다.
        return cov

    # 분산을 계산하는 메소드이다.
    def variance(
        self,
        X: list
    ) -> float:

        var = 0

        # "X"의 평균을 저장한다.
        sampleMean = self.mean(X)

        # 분산의 계산과정이다.
        for x in X:
            var += (x - sampleMean)**2

        # 원래 분산은 "X"의 크기로 나누지만,
        # 전체적인 계산량을 고려해 나누는 식을 제거했다.
        return var

    # 계수를 결정하는 함수이다.
    def train(
        self,
        X: list,
        Y: list,
    ):

        # "X", "Y"의 평균을 저장한다.
        xMean = self.mean(X)
        yMean = self.mean(Y)

        # 각 계수를 저장한다.
        # 계산 방법은 AI PLAYGROUND의 "모델, 최적화 가이드라인"을 참고
        self.W[1] = self.covariance(X, Y)/self.variance(X)
        self.W[0] = yMean - self.W[1]*xMean

    # 새로운 데이터에 대해,
    # 예측하는 메소드이다.
    def predict(
        self,
        x: float
    ) -> float:
        return self.W[0] + self.W[1]*x

    # 렌더링을 위한 초기화 메소드이다.
    def render_init(
        self,
        X: list,
        Y: list,
    ):

        # 학습 결과를 보여주기 전에,
        # 데이터를 먼저 플롯한다.
        self.ax.scatter(X, Y,
                        s=80,
                        color='blue',
                        marker='o')

    # 학습 결과를 플롯한다.
    def draw_line(
        self,
        X: list,
        Y: list,
        fn: str,  # 저장할 파일 이름
        line_color: str = "black",  # 회귀직선의 색
    ):

        # 어디까지 출력할 것인지,
        # 출력할 선의 크기를 결정한다.
        x_min = np.min(X) - 1
        x_max = np.max(X) + 1
        # "X"의 최소치 - 1, "X"의 최대치 + 1 사이를 출력한다.
        line_range = np.arange(x_min, x_max, 0.1)

        line_x = [x for x in line_range]  # x 축의 데이터
        line_y = [self.W[0] + x*self.W[1] for x in line_range]  # y 축의 데이터

        # 선을 플롯한다.
        self.ax.plot(line_x, line_y, color=line_color, label=fn)
        self.ax.legend()

        # 그래프를 저장한다.
        plt.savefig(f"./results/{fn}.png")


if __name__ == "__main__":
    X = [31.92, 33.55, 34.23, 34.97, 35.43, 36.38, 38.27]
    Y = [16.7, 15.9, 13.8, 14.8, 13.2, 12.2, 9.2]

    linear_model = LSM()
    linear_model.render_init(X, Y)

    linear_model.draw_line(X, Y, "before_train", line_color="red")

    linear_model.train(X, Y)

    linear_model.draw_line(X, Y, "after_train")

    plt.show()
