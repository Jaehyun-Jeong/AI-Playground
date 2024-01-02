from typing import Type, Tuple
import numpy as np


class Node:

    def __init__(
        self,
        feature: int = None,  # 피처의 인덱스, Branch 노드면 필요
        threshold: float = None,  # 임계값, Branch 노드면 필요
        classIdx: int = None  # 분류할 클래스, Leaf 노드면 필요
    ):

        self._left: Type[Node] = None  # 왼쪽 노드
        self._right: Type[Node] = None  # 오른쪽 노드

        # Branch 노드를 위한 인스턴스
        self.feature: int = feature
        self.threshold: float = threshold

        # Leaf 노드를 위한 인스턴스
        self.classIdx: int = classIdx

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, node):
        self._left = node

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, node):
        self._right = node

    # Leaf 노드인지, Branch 노드인지 확인
    def is_leaf(self) -> bool:
        return False if self.classIdx is None else True


class TreeBased():

    def __init__(
        self
    ):

        self._numFeatures: int = None
        self._root: Type(Node) = None

    # 각 클래스당 데이터 수를 dictionary 형태로 리턴한다.
    # ex)
    # {'dog': 30, 'cat', 21}
    @staticmethod
    def __count_class(
        Y: np.ndarray
    ) -> dict:

        uniqueClasses, countsClasses = np.unique(Y, return_counts=True)

        return dict(zip(uniqueClasses, countsClasses))

    @staticmethod
    def __select_class(
        Y: np.ndarray
    ):

        try:
            selectClass = np.bincount(Y).argmax()
        except ValueError:
            selectClass = None

        return selectClass

    @staticmethod
    def __split(
        X: np.ndarray,
        featureIdx: int,
        threshold: float
    ) -> Tuple[int, int]:

        XFeature = X[:, featureIdx]
        leftIdxs = np.argwhere(XFeature < threshold).flatten()
        rightIdxs = np.argwhere(XFeature >= threshold).flatten()

        return leftIdxs, rightIdxs

    @staticmethod
    def gini_impurity(
        Y: np.ndarray
    ) -> float:

        giniImpurity: float = 1
        listClasses: np.ndarray = np.unique(Y)
        sizeY: int = len(Y)

        for classIdx in listClasses:
            giniImpurity -= (np.sum(Y == classIdx) / sizeY)**2

        return giniImpurity
