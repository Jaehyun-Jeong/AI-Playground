import pandas as pd

df = pd.read_csv("./diamond_rock.csv")
temp_dict = {}
temp_list = []
label_num = len(df['sort'].unique())

for new_col in df['sort'].unique():

    temp_list = []

    for label in df['sort']:
        if label == new_col:
            temp_list.append(1)
        else:
            temp_list.append(0)

    temp_dict[new_col] = temp_list

one_hot_df = pd.DataFrame(data=temp_dict)


print(df)
df = df.drop('sort', axis=1)
df = pd.concat([df, one_hot_df], axis=1)
print("=============================================")
print(df)
