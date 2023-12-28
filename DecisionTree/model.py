from typing import Type
import math as ma
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

    @staticmethod
    def __select_class(
        Y: np.ndarray
    ):

        return np.bincount(Y).argmax()

    @staticmethod
    def __split(
        X: np.ndarray,
        featureIdx: int,
        threshold: float
    ):

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
        self._root = Node()

        # ex) cat, dog classifier with 100 data
        # [50, 50]: 50 data of cat, 50 data of dog
        # [100, 0]:
        self._classes: list = None

    @property
    def classes(self):
        return self._classes

    @classes.setter
    def classes(self, classes: list):
        self._classes = classes

    def is_classified(self):

        setClasses: set = set(self._classes)
        isClassified: bool = False

        if len(setClasses) == 2 \
                and self._classes.count(0) == (len(self._classes) - 1):

            isClassified = True

        return isClassified


class HumanTree(TreeBased):

    def __init__(
        self,
    ):

        super().__init__()

        self._currentNode: Type(HumanTreeNode) = None
        self._currentNodeIdx: int = None

    # 인덱싱은 다음과 같이 한다.
    #       1       depth: 0
    #      / \
    #     /   \
    #    2     3    depth: 1
    #   / \   / \
    #  4   5 6   7  depth: 2
    @staticmethod
    def __get_idx(
        depth: int,
        number: int
    ):

        return ma.pow(2, depth) + number

    def select_node(
        self,
        depth: int,
        number: int,
    ):

        self._currentNode = self._root

        # Create an order of Lefts and Rights
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

                if not self._currentNode.left:
                    raise ValueError(\"No node! First, create a node using the \"build_branch\" method")
                else:
                    self._currentNode = self._currentNode.left

            elif move == '1':  # Right

                if self._currentNode.right== None:
                    raise ValueError("No node! First, create a node using the \"build_branch\" method")
                else:
                    self._currentNode = self._currentNode.right

            else:
                raise Exception("An unexpected error occurred")

        self._currentNodeIdx = self.__get_idx(depth, number)

    def start_train(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ):

        pass

    def predict(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ):

        pass

    def build_branch(
        self,
        feature: int,
        threshold: float
    ):

        ans: str = None
        isBuild: bool = True

        if self._currentNode == None:
            raise ValueError("Select a node first using \"select_node\" method")

        # When node is already perfectly classied classes
        if len(self._currentNode.classes) == 1:

            ans = input("It's already perfectly classified, do you still want to do it? (y|n):")
            ans = ans.lower()

            # '', 'y', or 'Y'
            if ans != '' and ans != 'y':
                isBuild = False

        if isBuild:

            self._currentNode.feature = feature
            self._currentNode.threshold = threshold

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
    df = read_csv("../Datasets/diamond_rock_index.csv")

    X = df[['hardness', 'brightness']].to_numpy()
    Y = df['sort'].to_numpy()

    DT = DecisionTree()
    DT.train(X, Y)

    print(DT._root.__dict__)
    print(DT._root.left.__dict__)
    print(DT._root.right.__dict__)
