from model import LogisticRegression
from pandas import read_excel
import numpy as np


# Load Data
df = read_excel("../Datasets/snow_data_one_hot.xlsx")

n_features = 8
X = df[['precipitation', 'max_temp', 'min_temp', 'humidity', 'spring', 'summer', 'fall', 'winter']].to_numpy()
Y = df['snow'].to_numpy()
Y[Y == 'yes'] = '1'
Y[Y == 'no'] = '0'

X = X.astype(np.float64)
Y = Y.astype(np.int64)

DT = LogisticRegression(
    n_features=n_features,
    learning_rate=0.001)
DT.train(X, Y, 200)
