import numpy as np
import pandas as pd


'''
ENG:
Handle devide by zero
KOR:
0또는 "tol"보다 작은 수로 나눌 때 값을 조정
'''
def denominator(
    arr: np.ndarray,
    tol: float = None
) -> np.ndarray:

    # The tolerance provided by NumPy
    if tol is None:
        tol = np.finfo(arr.dtype).eps

    newArr: np.ndarray  = np.copy(arr)
    newArr[newArr == 0] = tol  # Replace 0 to "tol"

    tempArr: np.ndarray = np.copy(newArr[np.abs(newArr) < tol])

    newArr[np.abs(newArr) < tol] = (tempArr / np.linalg.norm(tempArr)) * tol

    return newArr

'''
ENG:
Adjust log probabilities when the probability is 0 or 1, as logarithmic operations are undefined for these values.
KOR:
log probability는 확룰이 1 또는 0 일 때 정의되지 않으므로, 값을 조정한다.
'''
def log_prob(
    prob: np.ndarray,
    tol: float = None
):

    newProb = np.copy(prob)

    # The tolerance provided by NumPy
    if tol is None:
        tol = np.finfo(newProb.dtype).eps

    # Clip probability from [0, 1] to [tol, 1 - tol]
    newProb = np.maximum(tol, np.minimum(1 - tol, newProb))

    return np.log(newProb)


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


'''
One hot encoding
parameters
    data: Data as pandas DataFrame type.
    textFeature: List of Feature names you want to vectorize.
    words: List of words that will be counted
'''
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
    df = one_hot(df, ['season'])
    print(df)

    df = pd.read_csv("./Datasets/positive_negative_sentences.tsv", sep='\t', header=0)
    print(df)
    df = bag_of_words(df, ["sentence"], ["좋아", "싫어"])
    print(df)
    '''
    arr = np.array([1.09023901e-13, 1.15463195e-14, 6.21724894e-15, 2.22044605e-16, 2.66453526e-15, 5.06261699e-14, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.68673972e-14, 1.77635684e-15, 4.44089210e-16, 6.66133815e-16, 2.46025422e-13])

    print(log_prob(arr))
