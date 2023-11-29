from model import LSM

import matplotlib.pyplot as plt
from pandas import read_excel

# Load Data
df = read_excel("../Datasets/height_data.xlsx")

X = df['age'].to_list()
Y = df['height'].to_list()

linear_model = LSM()
linear_model.render_init(X, Y)

# linear_model.draw_line(X, Y, "before_train", line_color="red")

linear_model.train(X, Y)

linear_model.draw_line(X, Y, "after_train")

plt.show()
