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

        # "self.classes"는 각 클레스에 대하여 얼마나 많은 데이터를 포함하는지 의미한다.
        # "self.classes" indicates how much data this node contains.
        # ex)
        # {'dog': 30, 'cat', 21}
        self.classes: dict = None


class HumanTree(TreeBased):

    def __init__(
        self,
    ):

        super().__init__()

        # "HumanTree"의 root 노드를 빈 노드로 초기화한다.
        self._root = Node()

        # 학습중인지를 알려주는 인스턴스 변수
        self._is_train: bool = False

        # 학습 데이터
        self.data: pd.DataFrame = None
        self.X: np.ndarray = None  # 학습 데이터
        self.Y: np.ndarray = None  # 타겟 데이터

        # "self._featureIdxMa"에 의해 인덱싱된 "Y"
        self.YIdx: np.ndarray = None  # Indexed Y by "self._featureIdxMap"

        # 지금 선택된 노드를 알려주는 인스턴스 변수
        self._currentNode: Type(HumanTreeNode) = None
        self._currentNodeIdx: int = None

        # Feature 이름에서 feature 인덱스로의 매핑
        self._featureIdxMap: dict = {}

        # 노드 인덱스에서 노드가 포함하는 데이터 인덱스들의 매핑
        # 이를 사용하여 특정 노드가 어떤 데이터를 포함하는지 알 수 있다.
        # ex)
        # self._nodeDataIdxs[0]는 루트 노드가 포함하는 데이터의 인덱스를 반환한다.
        # self._nodeDataIdxs[4]는 깊이 2의 2번째 노드에 들어있는 데이터의 엔덱스를 반환한다.
        # 노드의 이덱싱 방법은 "_get_idx" 메소드를 참고하자.
        self._nodeDataIdxs: dict = {}

    # 인덱싱은 다음과 같이 한다.
    #       0       depth: 0
    #      / \
    #     /   \
    #    1     2    depth: 1
    #   / \   / \
    #  3   4 5   6  depth: 2
    @staticmethod
    def _get_idx(
        depth: int,
        number: int
    ) -> int:

        return int(ma.pow(2, depth) + number - 1)

    # "number"인자 는 다음 트리의 숫자와 같이 부여한다.
    #       0       depth: 0
    #      / \
    #     /   \
    #    0     1    depth: 1
    #   / \   / \
    #  0   1 2   3  depth: 2
    def select_node(
        self,
        depth: int,  # 선택할 노드의 깊이 (0부터 시작한다.)
        number: int,  # 선택할 노드의 번호 (0부터 시작하고, 번호는 위같이 부여한다.)
    ):

        # "number"는 0 이상이고, 2의 깊이제곱 보다 작으므로, 범위를 벗어나면 에러를 출력한다.
        if number < 0 or number >= ma.pow(2, depth):
            raise ValueError(
                "\"number\" must be in the range of "
                "(0 <= \"number\" < 2 power of depth)")

        # 현재 노드의 정보를 수정한다.
        self._currentNodeIdx = self._get_idx(depth, number)  # 노드 인덱스를 수정한다.

        # Root 노드부터 시작해서 인자 "depth", "number"를 만족하는 노드를 찾는다.
        self._currentNode = self._root  # Root 노드부터 찾기 시작한다.

        # "depth"가 0 이면 Root 노드이므로 탐색을 하지 않는다.
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
            binNumber = format(number, 'b')  # number를 binary string으로 바꾼다.
            # 전체 깊이를 고려해 0(Left)를 추가한다.
            # "orderLeftRight"는 Root에서 이동할 순서의 리스트이다.
            orderLeftRight = '0'*(depth - len(binNumber)) + binNumber

            for move in orderLeftRight:

                if move == '0':  # Left

                    # 왼쪽에 노드가 없으면 에러를 출력한다.
                    if self._currentNode.left is None:
                        raise ValueError(
                            "No node! First, "
                            "create a node using the \"build_branch\" method")
                    else:
                        self._currentNode = self._currentNode.left

                elif move == '1':  # Right

                    # 오른쪽에 노드가 없으면 에러를 출력한다.
                    if self._currentNode.right is None:
                        raise ValueError(
                            "No node! First, "
                            "create a node using the \"build_branch\" method")
                    else:
                        self._currentNode = self._currentNode.right

                # '0', '1' 외의 문자가 들어있으면 에러를 출력한다.
                else:
                    raise Exception("An unexpected error occurred")

    # 학습을 시작하는 함수이다. 함수 시작과 동시에 root 노드를 선택한다.
    # 이 함수를 사용하기 전까지는 "build_branch" 메소드를 사용할 수 없다.
    def start_train(
        self,
        # 학습에 사용할 feature 이름의 list
        featureList: list,
        targetList: list,
        # 학습에 사용할 실제 데이터
        data: pd.DataFrame
    ):

        # Feature의 개수를 저장한다.
        self._numFeatures = len(featureList)

        # 데이터를 저장하고, "self.X", "self.Y"에 numpy.ndarray 형태로 저장한다.
        self.data = data
        self.X = data[featureList].to_numpy()
        self.Y = data[targetList].to_numpy().flatten()

        # 학습을 시작했으므로, "self._is_train"을 True로 바꾼다.
        self._is_train = True

        nData: int = self.X.shape[0]  # 데이터의 수

        # 클래스를 인덱스로 바꿔서 "self.YIdx"에 저장한다.
        self.YIdx = np.zeros(self.Y.shape, dtype=int)
        # Feature에서 인덱스로의 맵 ("self._featureIdxMap")을 만들면서,
        # "self.Y"를 인덱싱한 "self.YIdx"를 만든다.
        for featureIdx, featureName in enumerate(featureList):
            self._featureIdxMap[featureName] = featureIdx
            self.YIdx[self.Y == featureName] = featureIdx

        # 학습을 시작하면 Root 노드를 선택
        self.select_node(depth=0, number=0)
        # root 노드가 포함하는 각 클래스의 데이터 수를 저장한다.
        # ex)
        # 전체 데이터에서 class가 'dog'인 데이터가 30개, 'cat'인 데이터가 21개인 경우
        # {'dog': 30, 'cat', 21}
        self._root.classes = self._TreeBased_count_class(self.Y)
        # Root 노드가 포함하는 데이터는 모든 데이터이므로,
        # 모든 데이터의 인덱스를 "self._nodeDataIdxs"에 저장한다.
        self._nodeDataIdxs[self._currentNodeIdx] = \
            np.array([i for i in range(nData)])

        # Root 노드의 예측 결과를 저장한다.
        self._root.classIdx = self._TreeBased_select_class(self.YIdx)

    # Feature의 이름과 "threshold"로 가지를 만든다.
    # 이는 먼저 "start_train" 메소드를 실행해야 사용할 수 있다.
    def build_branch(
        self,
        featureName: str,  # 조건으로 만들 feature의 이름
        threshold: float  # 조건으로 만들 feature의 threshold
    ):

        # Error Handling
        # 학습중이 아닌 경우, "start_train" 메소드를 먼저 실행해야한다.
        if not self._is_train:
            raise ValueError(
                "Start training before create new branches "
                "using the \"start_train\" method")
        # 선택된 노드가 없다면 먼저 선택해야 한다.
        if self._currentNode is None:
            raise ValueError(
                "Select a node first "
                "using \"select_node\" method")

        # Feature 이름을 feature의 인덱스로 바꾼다.
        featureIdx = self._featureIdxMap[featureName]

        # "featureIdx"와 "threshold"를 기준으로 데이터를 나눈다.
        # 하지만 데이터를 나눠서 저장하면 메모리 손실이 크다.
        # 따라서 데이터의 인덱스만을 나눠서 저장한다.
        parentIdxs = self._nodeDataIdxs[self._currentNodeIdx]
        leftIdxs, rightIdxs = \
            self._TreeBased_split(self.X, featureIdx, threshold)
        leftIdxs = list(set(parentIdxs).intersection(leftIdxs))
        rightIdxs = list(set(parentIdxs).intersection(rightIdxs))

        # 왼쪽, 오른쪽 자식노드를 만든다.
        # 이때, 현재 노드의 번호가 n 이라면 다음과 같이 자식노드의 인덱스를 계산한다.
        # left child: 2*n + 1
        # right child: 2*n + 2
        self._nodeDataIdxs[2*self._currentNodeIdx + 1] = leftIdxs
        self._nodeDataIdxs[2*self._currentNodeIdx + 2] = rightIdxs

        # 왼쪽, 오른쪽 자식노드의 분류 결과를 저장한다.
        leftClassIdx = self._TreeBased_select_class(self.YIdx[leftIdxs])
        rightClassIdx = self._TreeBased_select_class(self.YIdx[rightIdxs])

        # 현재 노드의 왼쪽, 오른쪽 자식 노드를 저장한다.
        self._currentNode.left = Node(classIdx=leftClassIdx)
        self._currentNode.right = Node(classIdx=rightClassIdx)

        # 현재 노드의 조건을 저장한다.
        self._currentNode.threshold = threshold
        self._currentNode.feature = featureIdx

        # 왼쪽, 오른쪽 자식 노드가 포함하는 클래스의 수를 저장한다.
        self._currentNode.left.classes = \
            self._TreeBased_count_class(self.Y[leftIdxs])
        self._currentNode.right.classes = \
            self._TreeBased_count_class(self.Y[rightIdxs])

    # 선택한 노드애 들어있는 데이터를 pandas.DataFrame 형태로 반환한다.
    def show(self) -> pd.DataFrame:
        return self.data.iloc[self._nodeDataIdxs[self._currentNodeIdx]]
