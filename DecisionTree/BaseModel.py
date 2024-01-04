from typing import Type, Tuple
import numpy as np


# 트리의 노드를 표현하는 클래스
class Node:

    def __init__(
        self,
        feature: int = None,  # 피처의 인덱스, Branch 노드면 필요하다.
        threshold: float = None,  # 임계값, Branch 노드면 필요하다.
        classIdx: int = None  # 분류할 클래스, Leaf 노드면 필요하다.
    ):

        self.left: Type[Node]  # 왼쪽 노드
        self.right: Type[Node]  # 오른쪽 노드

        # Branch 노드를 위한 인스턴스 변수
        self.feature: int = feature
        self.threshold: float = threshold

        # Leaf 노드를 위한 인스턴스 변수
        self.classIdx: int = classIdx

    # Leaf 노드인지, Branch 노드인지 확인한다.
    def is_leaf(self) -> bool:
        return False if self.classIdx is None else True


# 모든 Tree의 부모 클래스
class TreeBased():

    def __init__(
        self
    ):

        self._numFeatures: int = None  # 피쳐의 수
        self._root: Type(Node) = None  # 루트 노드

    # 각 클래스당 데이터 수를 dictionary 형태로 리턴한다.
    # ex)
    # {'dog': 30, 'cat', 21}
    @staticmethod
    def __count_class(
        Y: np.ndarray
    ) -> dict:

        uniqueClasses, countsClasses = np.unique(Y, return_counts=True)

        return dict(zip(uniqueClasses, countsClasses))

    # 'Y'에서 가장 많은 클래스의 인덱스를 리턴한다.
    @staticmethod
    def __select_class(
        Y: np.ndarray
    ) -> np.int_:

        # 'Y'에서 가장 많은 인덱스를 "selectClass"에 저장한다.
        selectClass: np.int

        try:
            selectClass = np.bincount(Y).argmax()
        except ValueError:
            selectClass = None

        return selectClass

    # 인덱스가 'featureIdx'인 feature의 'threshold'를 기준으로 데이터를 나눈다.
    @staticmethod
    def __split(
        X: np.ndarray,
        featureIdx: int,
        threshold: float
    ) -> Tuple[list[int], list[int]]:

        # feature를 고른다.
        XFeature = X[:, featureIdx]

        # "threshold"를 기준으로한 X의 인덱스 리스트를 만든다.
        leftIdxs = np.argwhere(XFeature < threshold).flatten()
        rightIdxs = np.argwhere(XFeature >= threshold).flatten()

        return leftIdxs, rightIdxs

    # 지니계수를 계산한다.
    @staticmethod
    def gini_impurity(
        Y: np.ndarray
    ) -> float:

        # 지니계수는 1에서 빼는 순서로 계산되므로 1로 초기화한다.
        giniImpurity: float = 1  # 지니계수
        listClasses: np.ndarray = np.unique(Y)  # 클래스 종류의 배열
        sizeY: int = len(Y)  # 'Y'의 크기

        # 지니계수를 계산한다.
        for classIdx in listClasses:
            giniImpurity -= (np.sum(Y == classIdx) / sizeY)**2

        return giniImpurity
