from BaseModel import Node, TreeBased
import numpy as np


class DecisionTree(TreeBased):

    def __init__(
        self,
    ):

        super().__init__()

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ):

        self._numFeatures = X.shape[1]
        featureIdxs = [i for i in range(self._numFeatures)]
        self._root = self.__build_tree(X, Y, featureIdxs)

    # 재귀적으로 나무를 생성
    def __build_tree(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        featureIdxs: list
    ):

        classes = np.unique(Y)
        featureIdx, threshold = \
            self.__best_criteria(X, Y, featureIdxs)

        # 아래의 조건을 만족하면 트리 생성 종료한다.
        # 조건1: 나눠진 노드 안에 있는 클래스의 수가 1이다.
        # 조건2: 나눌 수 있는 threshold, feature가 존재하지 않는다.
        if len(classes) == 1 or not (featureIdx or threshold):
            # Leaf 노드를 리턴한다.
            return Node(classIdx=self._TreeBased__select_class(Y))

        leftIdxs, rightIdxs = self._TreeBased__split(X, featureIdx, threshold)

        left = self.__build_tree(
            X[leftIdxs, :],
            Y[leftIdxs],
            featureIdxs)
        right = self.__build_tree(
            X[rightIdxs, :],
            Y[rightIdxs],
            featureIdxs)

        # Root 노드를 생성한다.
        node = Node(feature=featureIdx, threshold=threshold)
        node.left = left
        node.right = right

        # Root 노드를 반환한다.
        return node

    def __best_criteria(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        featureIdxs: list
    ):

        bestFeature: int = None
        bestThreshold: float = None
        giniImpurity: float = None
        bestGiniImpurity: float = 1  # 1은 가장 안좋은 지니계수이다.
        size: int = len(Y)

        for featureIdx in featureIdxs:
            XFeature = X[:, featureIdx]
            uniqueValues = np.unique(XFeature)

            # If "XFeature" has one unique value,
            # no threshold
            if len(uniqueValues) != 1:
                thresholds = self.__thresholds(uniqueValues)
                for threshold in thresholds:
                    leftIdxs, rightIdxs = \
                        self._TreeBased__split(X, featureIdx, threshold)
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

        return bestFeature, bestThreshold

    @staticmethod
    def __thresholds(
        uniqueValues: np.ndarray
    ):

        size: int = len(uniqueValues)
        thresholds: list = [0] * (size-1)

        for i in range(size-1):
            thresholds[i] = (uniqueValues[i] + uniqueValues[i+1]) / 2

        return thresholds
