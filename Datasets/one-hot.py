import pandas as pd

df = pd.read_excel("./snow_data.xlsx")
featureName = 'season'
temp_dict = {}
temp_list = []
label_num = len(df[featureName].unique())

for new_col in df[featureName].unique():

    temp_list = []

    for label in df[featureName]:
        if label == new_col:
            temp_list.append(1)
        else:
            temp_list.append(0)

    temp_dict[new_col] = temp_list

one_hot_df = pd.DataFrame(data=temp_dict)


print(df)
df = df.drop(featureName, axis=1)
df = pd.concat([df, one_hot_df], axis=1)
print("=============================================")
print(df)
df.to_excel("./snow_data_one_hot.xlsx")
