import sys
sys.path.append("../") # to import module

from model import LinearModel
from utils import bag_of_words

import pandas as pd
import numpy as np

df = pd.read_csv("../Datasets/positive_negative_sentences.tsv", sep='\t', header=0)
df = bag_of_words(df, ["sentence"], ["좋아", "싫어"])

X = df[['좋아', '싫어']].to_numpy()
y = df['label'].map({'positive': 1, 'negative': 0}).to_numpy()

linear_model = LinearModel(learning_rate=0.05)
linear_model.train(X, y, 100)
