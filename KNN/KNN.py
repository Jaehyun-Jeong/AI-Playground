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
        norm = list(self._norm(x))

        # 거리의 인덱스를 거리 순으로 정렬한다.
        # ex)
        # 거리: [2.0, 4.0, 4.0, 4.0, 4.0, 3.0, 4.0, 4.0, 6.0, 3.0, 5.0, 5.0]
        # 인덱스: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        # 실행 후
        # 거리: [2.0, 4.0, 4.0, 4.0, 4.0, 3.0, 4.0, 4.0, 6.0, 3.0, 5.0, 5.0]
        # 인덱스: [0, 5, 9, 1, 2, 3, 4, 6, 7, 10, 11, 8]
        sortedNormIdxs = self.sort_idxs(norm, sortedNormIdxs)

        # 인덱스를 참조하여 가장 많은 수의 클래스를 출력한다.
        return np.bincount(self.Y[sortedNormIdxs[:self.k]]).argmax()

    # Quick sort 를 사용해서 정렬한다.
    def sort_idxs(
        self,
        # X 를 소트하고, 바뀐 순서를 idxs에 저장한다.
        values: list,
        idxs: list,
    ) -> list:

        pivot: float  # Quick Sort 의 pivot

        leftValues: list = []
        rightValues: list = []
        leftIdxs: list = []
        rightIdxs: list = []

        if len(values) <= 1:
            return idxs
        else:

            # pivot 으로 0 번째 숫자를 선택한다.
            pivot = values[0]

            for idx, value in zip(idxs[1:], values[1:]):
                if value < pivot:
                    leftValues.append(value)
                    leftIdxs.append(idx)
                else:
                    rightValues.append(value)
                    rightIdxs.append(idx)

            return self.sort_idxs(leftValues, leftIdxs) + [idxs[0]] + \
                self.sort_idxs(rightValues, rightIdxs)


if __name__ == "__main__":
    from pandas import read_csv

    df = read_csv("../Datasets/zebra_giraffe_crowd_noise_index.csv")
    X = df[['x', 'y']].to_numpy()
    Y = df['species'].to_numpy()

    model = KNN(X, Y, k=1, p=1)
    print(model.predict([0, 0]))
