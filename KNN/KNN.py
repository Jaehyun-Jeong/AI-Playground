import numpy as np


# KNN 의 클래스이다.
class KNN:

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ):

        # 데이터를 입력받는다.
        self.X: np.ndarray = X
        self.Y: np.ndarray = Y

    def _norm(
        self,
        x: np.ndarray  # 2d array 로 넣어야 한다.
    ):

        # 거리를 계산을 위해 차이를 먼저 구한다.
        diff: np.ndarray = np.zeros(self.X.shape)

        for idx, row in enumerate(self.X):
            diff[idx] = x - row

        print(diff)


if __name__ == "__main__":
    pass
