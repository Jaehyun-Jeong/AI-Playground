import numpy as np


# KNN 의 클래스이다.
class KNN:

    def __init__(
        self,
        # 예측에 사용할 데이터
        X: np.ndarray,
        Y: np.ndarray,
        k: int,  # K Nearest Neighbor 에서의 K
        p: int == 2  # p-norm의 p
    ):

        # 데이터를 입력받는다.
        self.X: np.ndarray = X
        self.Y: np.ndarray = Y

        self.k = k
        self._p = p

    def _norm(
        self,
        X: np.ndarray,  # 1d array 로 넣어야 한다.
    ):

        # 에러 핸들링
        if len(X.shape) != 1:
            raise ValueError("\"X\" must be the 1d array!")

        # norm을 저장하기 위한 변수
        norm: np.ndarray = np.zeros(self.X.shape)

        # 거리를 계산을 위해 차이를 먼저 구한다.
        for idx, row in enumerate(self.X):
            norm[idx] = X - row

        # p-norm 을 계산한다.
        norm = np.sum(np.abs(norm)**self._p, axis=-1)**(1./self._p)

        return norm

    def predict(
        self,
        x: list  # 1d array 로 넣어야 한다.
    ):

        norm: np.ndarray  # 거리를 저장
        # 소트 후의 순서를 저장한다.
        sortedNormIdxs: list = [i for i in range(self.X.shape[0])]

        # list 데이터를 np.ndarray 로 변환한다.
        x = np.array(x, dtype=self.X.dtype)

        # 에러 핸들링
        if len(x.shape) != 1:
            raise ValueError("\"X\" must be the 1d array!")

        # 거리를 구한다.
        norm = self._norm(x)

        sortedNormIdxs = self.sort_idxs(list(norm), sortedNormIdxs)

    # Quick sort 를 사용해서 정렬한다.
    @staticmethod
    def sort_idxs(
        # X 를 소트하고, 바뀐 순서를 idxs에 저장한다.
        values: list,
        idxs: list
    ):

        pivot: float  # Quick Sort 의 pivot

        if(1 < len(values)):
            pivot = values[0]
            pass

if __name__ == "__main__":
    from pandas import read_csv

    df = read_csv("../Datasets/zebra_giraffe_crowd_index.csv")
    X = df[['x', 'y']].to_numpy()
    Y = df['species'].to_numpy()

    model = KNN(X, Y, k=3, p=1)
    model.predict(list(X[0]))
