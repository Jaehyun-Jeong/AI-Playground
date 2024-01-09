from typing import Type
import numpy as np

from BaseModel import Node, TreeBased


# 결정 트리 클래스
class DecisionTree(TreeBased):

    def __init__(
        self,
    ):

        super().__init__()

    def train(
        self,
        X: np.ndarray,  # 학습 데이터
        Y: np.ndarray  # 학습 타겟
    ):

        # Feature의 수는 'X'의 열 수이다.
        self._numFeatures = X.shape[1]  # Feature의 수

        # Feature를 인덱싱
        featureIdxs: list = [i for i in range(self._numFeatures)]
        self._root = self._build_tree(X, Y, featureIdxs)  # Root 노드 생성

    # 재귀적으로 나무를 생성
    def _build_tree(
        self,
        X: np.ndarray,  # 학습 데이터
        Y: np.ndarray,  # 학습 타겟
        featureIdxs: list  # Feature의 인덱스
    ) -> Type(Node):

        # 클래스의 종류를 확인한다.
        classes = np.unique(Y)
        # 지니계수를 고려해 feature와 "threshold"을 찾는다.
        featureIdx, threshold = \
            self._best_criteria(X, Y, featureIdxs)

        # 아래의 조건을 만족하면 트리 생성 종료한다.
        # 조건1: 나눠진 노드 안에 있는 클래스의 수가 1이다.
        # 조건2: 나눌 수 있는 threshold, feature가 존재하지 않는다.
        if len(classes) == 1 or not (featureIdx or threshold):
            # Leaf 노드를 리턴한다.
            return Node(classIdx=self._TreeBased_select_class(Y))

        # 선택된 feature와 "threshold"를 기준으로 데이터를 나눈다.
        # 데이터의 인덱스를 나눠서, "leftIdxs", "rightIdxs"에 저장한다.
        leftIdxs, rightIdxs = self._TreeBased_split(X, featureIdx, threshold)

        # 나눠진 데이터에 대해, 나무를 계속 만든다.
        left = self._build_tree(
            X[leftIdxs, :],
            Y[leftIdxs],
            featureIdxs)
        right = self._build_tree(
            X[rightIdxs, :],
            Y[rightIdxs],
            featureIdxs)

        # Root 노드를 생성한다.
        node = Node(feature=featureIdx, threshold=threshold)
        node.left = left
        node.right = right

        # Root 노드를 반환한다.
        return node

    # 데이터
    def _best_criteria(
        self,
        X: np.ndarray,  # 학습 데이터
        Y: np.ndarray,  # 학습 타겟
        featureIdxs: list  # Feature의 인덱스
    ):

        # 지니계수에 기반하여 가장 좋은 feature를 저장한다.
        bestFeature: int = None
        # 지니계수와 "bestFeature"에 기반하여 가장 좋은 "threshold"를 저장한다.
        bestThreshold: float = None
        # 계산한 지니계수를 저장한다.
        giniImpurity: float = None
        # 가장 낮은(좋은) 지니계수를 저장한다.
        bestGiniImpurity: float = 1  # 1은 가장 안좋은 지니계수이다.
        # 데이터의 수
        size: int = len(Y)  # 지니계수 계산에 쓰인다.

        # 지니계수가 낮은 feature를 찾기 위해 모든 feature를 탐색한다.
        for featureIdx in featureIdxs:

            # Feature에서 유일한 값들을 찾는다.
            XFeature: np.ndarray = X[:, featureIdx]
            uniqueValues: np.ndarray = np.unique(XFeature)

            # 만약 feature의 유일한 값이 하나라면, 나눌 수 없다.
            if len(uniqueValues) != 1:

                # 나눌 수 있는 "threshold"의 리스트를 만든다.
                # ex)
                # 만약 feature에 유일한 값이 [1, 2, 3]이라면,
                # "thresholds"는 [1.5, 2.5]가 된다.
                thresholds = self._thresholds(uniqueValues)

                # 어느 "threshold"를 기준으로 나누면,
                # 지니계수가 가장 낮아지는지 찾는다.
                for threshold in thresholds:

                    # "threshold"를 기준으로 데이터를 나눈다.
                    leftIdxs, rightIdxs = \
                        self._TreeBased_split(X, featureIdx, threshold)

                    # 왼쪽, 오른쪽 노드의 지니계수를 계산한다.
                    leftGiniImpurity = self.gini_impurity(Y[leftIdxs])
                    rightGiniImpurity = self.gini_impurity(Y[rightIdxs])

                    # 가중 지니계수를 계산한다.
                    sizeLeft = len(leftIdxs)
                    sizeRight = len(rightIdxs)
                    giniImpurity = 0
                    giniImpurity += \
                        leftGiniImpurity * sizeLeft / size
                    giniImpurity += \
                        rightGiniImpurity * sizeRight / size

                    # 지니 계수는 작을수록 균일도가 높으므로,
                    # "bestGiniImpurity"를 다음과 같이 수정한다.
                    if bestGiniImpurity > giniImpurity:
                        bestGiniImpurity = giniImpurity
                        bestFeature = featureIdx
                        bestThreshold = threshold

        # 가장 지니계수를 낮게(좋게) 만드는 feature와 threshold를 반환한다.
        return bestFeature, bestThreshold

    # Feature에 있는 값들을 기준으로,
    # Threshold 후보 리스트를 만든다.
    @staticmethod
    def _thresholds(
        uniqueValues: np.ndarray  # Feature가 가지는 값들
    ):

        # Feature가 가지는 유일한 값들의 크기
        size: int = len(uniqueValues)
        # 후보의 수는 size - 1 이다.
        # ex) Feature의 값들이 [1, 3, 4](size == 3)로 이루어지면,
        # "thresholds"는 [2, 3.5](size == 2) 이다.
        # 따라서 "thresholds"의 크기는 "size" - 1 이다.
        thresholds: list = [0] * (size-1)

        # Threshold의 값을 계산한다.
        for i in range(size-1):
            thresholds[i] = (uniqueValues[i] + uniqueValues[i+1]) / 2

        # 계산한 "thresholds"를 반환한다.
        return thresholds
