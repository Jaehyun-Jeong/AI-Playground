from typing import Type, Tuple
import math as ma
import numpy as np
import pandas as pd


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
        leftIdxs = np.argwhere(XFeature <= threshold).flatten()
        rightIdxs = np.argwhere(XFeature > threshold).flatten()

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


class HumanTreeNode(Node):

    def __init__(
        self
    ):

        super().__init__()

        # "self._classes" indicates how much data this node contains.
        # ex)
        # {'dog': 30, 'cat', 21}
        self._classes: dict = None

    @property
    def classes(self):
        return self._classes

    @classes.setter
    def classes(self, classes: dict):
        self._classes = classes


class HumanTree(TreeBased):

    def __init__(
        self,
    ):

        super().__init__()
        self._root = Node()

        # Check it's training
        self._is_train: bool = False

        # Training data
        self.X: pd.DataFrame = None
        self.Y: pd.DataFrame = None
        self.YIdx: np.ndarray = None  # Indexed Y by "self._featureIdxMap"

        # Indicates a selected node
        self._currentNode: Type(HumanTreeNode) = None
        self._currentNodeIdx: int = None

        # Mapping from feature to feature index
        self._featureIdxMap: dict = {}

        # node index to data indexes
        self._nodeDataIdxs: dict = {}

    # 인덱싱은 다음과 같이 한다.
    #       0       depth: 0
    #      / \
    #     /   \
    #    1     2    depth: 1
    #   / \   / \
    #  3   4 5   6  depth: 2
    @staticmethod
    def __get_idx(
        depth: int,
        number: int
    ) -> int:

        return int(ma.pow(2, depth) + number - 1)

    # Numbers are
    #       0       depth: 0
    #      / \
    #     /   \
    #    0     1    depth: 1
    #   / \   / \
    #  0   1 2   3  depth: 2
    def select_node(
        self,
        depth: int,
        number: int,
    ):

        if number < 0 or number >= ma.pow(2, depth):
            raise ValueError(
                "\"number\" must be in the range of "
                "(0 <= \"number\" < 2 power of depth)")

        self._currentNodeIdx = self.__get_idx(depth, number)
        self._currentNode = self._root

        # If depth == 0, just return the root
        if depth != 0:

            # "depth", "number"의 정보만으로,
            # root에서 어떤 순서로 움직일지 결정한다.
            # ex)
            # depth == 3
            # 0: 000 (LLL) => Left->Left->Left
            # 1: 001 (LLR) => Left->Left->Right
            # 2: 010 (LRL) => Left->Right->Left
            # 3: 011 (LRR) => Left->Right->Right
            # 4: 100 (RLL) => Right->Left->Left
            # 5: 101 (RLR) => Right->Left->Left
            # 6: 110 (RRL) => Right->Right->Left
            # 7: 111 (RRR) => Right->Right->Right
            binNumber = format(number, 'b')
            orderLeftRight = '0'*(depth - len(binNumber)) + binNumber

            for move in orderLeftRight:

                if move == '0':  # Left

                    if self._currentNode.left is None:
                        raise ValueError(
                            "No node! First, "
                            "create a node using the \"build_branch\" method")
                    else:
                        self._currentNode = self._currentNode.left

                elif move == '1':  # Right

                    if self._currentNode.right is None:
                        raise ValueError(
                            "No node! First, "
                            "create a node using the \"build_branch\" method")
                    else:
                        self._currentNode = self._currentNode.right

                else:
                    raise Exception("An unexpected error occurred")

    def start_train(
        self,
        featureList: list,
        targetList: list,
        data: pd.DataFrame
    ):

        self._numFeatures = len(featureList)

        self.X = data[featureList].to_numpy()
        self.Y = data[targetList].to_numpy().flatten()
        self._is_train = True

        nData: int = self.X.shape[0]
        self.YIdx = np.zeros(self.Y.shape, dtype=int)  # class as index

        # Create a map from feature to featureIdx
        for featureIdx, featureName in enumerate(featureList):
            self._featureIdxMap[featureName] = featureIdx
            self.YIdx[self.Y == featureName] = featureIdx  # Set "self.YIdx"

        # Select root node
        self.select_node(depth=0, number=0)
        self._root.classes = self._TreeBased__count_class(self.Y)
        self._nodeDataIdxs[self._currentNodeIdx] = \
            np.array([i for i in range(nData)])

        # Select root node classifing result
        self._root.classIdx = self._TreeBased__select_class(self.YIdx)

    def build_branch(
        self,
        featureName: str,
        threshold: float
    ):

        # Error Handling
        if not self._is_train:
            raise ValueError(
                "Start training before create new branches "
                "using the \"start_train\" method")
        if self._currentNode is None:
            raise ValueError(
                "Select a node first "
                "using \"select_node\" method")

        # Convert feature name to feature index
        featureIdx = self._featureIdxMap[featureName]

        # Split data
        parentIdxs = self._nodeDataIdxs[self._currentNodeIdx]
        leftIdxs, rightIdxs = \
            self._TreeBased__split(self.X, featureIdx, threshold)
        leftIdxs = list(set(parentIdxs).intersection(leftIdxs))
        rightIdxs = list(set(parentIdxs).intersection(rightIdxs))

        # child index formulas of binary tree are
        # left child: 2*n + 1
        # right child: 2*n + 2
        self._nodeDataIdxs[2*self._currentNodeIdx + 1] = leftIdxs
        self._nodeDataIdxs[2*self._currentNodeIdx + 2] = rightIdxs

        # Left and right nodes classifing results
        leftClassIdx = self._TreeBased__select_class(self.YIdx[leftIdxs])
        rightClassIdx = self._TreeBased__select_class(self.YIdx[rightIdxs])

        self._currentNode.threshold = threshold
        self._currentNode.feature = featureIdx

        self._currentNode.left = Node(classIdx=leftClassIdx)
        self._currentNode.right = Node(classIdx=rightClassIdx)

        self._currentNode.left.classes = \
            self._TreeBased__count_class(self.Y[leftIdxs])


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
            return Node(classIdx=self.__select_class(Y))

        leftIdxs, rightIdxs = self.__split(X, featureIdx, threshold)

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
                        self.__split(X, featureIdx, threshold)
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


if __name__ == "__main__":

    from pandas import read_csv

    # Load Data
    df = read_csv("../Datasets/diamond_rock.csv")

    featureList = ['hardness', 'brightness']
    targetList = ['sort']

    HT = HumanTree()
    HT.start_train(featureList, targetList, df)

    print(HT._featureIdxMap)
