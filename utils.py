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
        # ex) # Remove 'season' => Add 'spring', 'summer', 'autumn', 'winter'
        del oneHotData[featureName]

        # Loop for unique values in "data".
        for newFeature in data[featureName].unique():
            newFeatureData = {}
            for dataIdx, label in enumerate(data[featureName]):
                if label == newFeature:
                    newFeatureData[dataIdx] = 1
                else:
                    newFeatureData[dataIdx] = 0

            oneHotData[f"{featureName}_{newFeature}"] = newFeatureData

    return pd.DataFrame(data=oneHotData)


# Count word from string
def count_word(string: str, word: str) -> int:

    lenWord = len(word)
    lenString = len(string)
    count = 0

    for i in range(lenString - lenWord + 1):
        if string[i : i + lenWord] == word:
            count += 1

    return count


def bag_of_words(
    data: pd.DataFrame,
    textFeature: list,
    words: list
):

    # Copy
    vectorizedData = data.to_dict()

    for featureName in textFeature:
        
        # Convert text feature to vectorized feature
        del vectorizedData[featureName]

        # Loop for each words
        for word in words:
            vectorizedFeatureData = {}
            for dataIdx, string in enumerate(data[featureName]):
                vectorizedFeatureData[dataIdx] = count_word(string, word)

            vectorizedData[word] = vectorizedFeatureData

    return pd.DataFrame(data=vectorizedData)


if __name__ == "__main__":

    '''
    df = pd.read_excel("./Datasets/snow_data.xlsx")
    df = one_hot(df, ['season', 'snow'])
    print(df)
    '''

    df = pd.read_csv("./Datasets/positive_negative_sentences.tsv", sep='\t', header=0)
    print(df)
    df = bag_of_words(df, ["sentence"], ["좋아", "싫어"])
    print(df)
