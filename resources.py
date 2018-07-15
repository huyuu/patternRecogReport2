import pandas as pd
import numpy as nu
from sklearn.decomposition import PCA
from matplotlib import pyplot as pl


# Note: - functions

def readDataFrom(FILES, columnNames):
    if len(FILES) != len(columnNames):
        return None
    columns = [ pd.read_csv(file, sep=' ', header=None).dropna(axis=1) for file, name in zip(FILES, columnNames) ]
    return pd.concat(columns, axis=1, keys=columnNames)


def euclideanMethod(data, testData, weight=pd.Series(nu.ones(256)) ):
    results = {}
    tf = testData['test'].loc[:, 0:255] # drop the last digit, which represent the correct class.
    for classNum in set(data.columns.get_level_values(0)): # equals: for str in ['a', 'ba', 'ka, 'pa']
        df = data[classNum]
        results[classNum] = (weight*(tf - df.mean())**2).sum(axis=1)
    return pd.DataFrame(results).idxmin(axis=1)


def similarityMethod(data, testData):
    tf = testData['test'].loc[:, 0:255] # drop the last digit, which represent the correct class.
    results = {}
    for classNum in set(data.columns.get_level_values(0)):
        mean = data[classNum].mean()
        cosThetas = (tf*mean).sum(axis=1) / tf.sum(axis=1)**0.5 / (mean**2).sum()**0.5
        results[classNum] = abs(cosThetas - 1)
    return pd.DataFrame(results).idxmin(axis=1)


def pcaMethod(data, testData):
    colors = ['r', 'k', 'y', 'b']
    # for classNum in set(data.columns.get_level_values(0)):
    #     pro = PCA(2).fit_transform(data[classNum])
    #     pl.scatter(pro[:, 0], pro[:, 1], color=colors.pop(), label=classNum)
    #
    # pl.legend()
    # pl.show()
    tf = testData['test'].loc[:, 0:255] # drop the last digit, which represent the correct class.
    results = {}
    pca = PCA(256)

    for classNum in set(data.columns.get_level_values(0)):
        pca.fit(data[classNum])
        projection = pca.transform(data[classNum])
        mean = pd.DataFrame(projection).mean()
        tfProjection = pd.DataFrame(pca.transform(tf))
        results[classNum] = ((tfProjection - mean)**2).sum(axis=1)
        
    return pd.DataFrame(results).idxmin(axis=1)
