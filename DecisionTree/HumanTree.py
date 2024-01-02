from BaseModel import Node, TreeBased

from typing import Type
import math as ma
import numpy as np
import pandas as pd


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
