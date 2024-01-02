from pandas import read_csv
from DecisionTree import DecisionTree

# Load Data
df = read_csv("../Datasets/diamond_rock_index.csv")

X = df[['hardness', 'brightness']].to_numpy()
Y = df['sort'].to_numpy()

DT = DecisionTree()
DT.train(X, Y)

print(DT._root.__dict__)
print(DT._root.left.__dict__)
print(DT._root.right.__dict__)
