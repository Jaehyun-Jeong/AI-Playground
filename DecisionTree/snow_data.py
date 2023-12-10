from pandas import read_excel
from model import DecisionTree

# Load Data
df = read_excel("../Datasets/snow_data.axcel")

X = df[['x', 'y']].to_numpy()
Y = df['species'].to_numpy()

DT = DecisionTree()
DT.train(X, Y)

print(DT._root.__dict__)
print(DT._root.left.__dict__)
print(DT._root.right.__dict__)
