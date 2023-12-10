from model import DecisionTree
from pandas import read_excel
import numpy as np


# Load Data
df = read_excel("../Datasets/snow_data_one_hot.xlsx")

X = df[['spring', 'summer', 'fall', 'winter']].to_numpy()
Y = df['snow'].to_numpy()
Y[Y == 'yes'] = '1'
Y[Y == 'no'] = '0'

X = X.astype(np.float64)
Y = Y.astype(np.int64)

DT = DecisionTree()
DT.train(X, Y)

print(DT._root.__dict__)
print(DT._root.left.__dict__)
print(DT._root.right.__dict__)
