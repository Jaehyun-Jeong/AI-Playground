import pandas as pd
from copy import copy


'''
One hot encoding
parameters
    data: Data as pandas DataFrame type.
    nominalFeature: List of Feature names to be encoded.
'''
def one_hot(
    data: pd.DataFrame,
    nominalFeatures: list
) -> pd.DataFrame:

    # Copy
    oneHotData = data.to_dict()

    # Loop for features to be one hot encoded.
    for featureName in nominalFeatures:

        # Remove original feature
        # ex) 
        # Remove 'season' => Add 'spring', 'summer', 'autumn', 'winter'
        del oneHotData[featureName]

        # Loop for unique values in "data".
        for newFeature in data[featureName].unique():
            newFeatureData = {}
            for dataIdx, label in enumerate(df[featureName]):
                if label == newFeature:
                    newFeatureData[dataIdx] = 1
                else:
                    newFeatureData[dataIdx] = 0

            oneHotData[newFeature] = newFeatureData

    print(oneHotData)

    return pd.DataFrame(data=oneHotData)


if __name__ == "__main__":

    df = pd.read_excel("./Datasets/snow_data.xlsx")
    df = one_hot(df, ['season'])
    print(df)

'''
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
'''
