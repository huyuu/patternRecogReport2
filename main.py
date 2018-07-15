from resources import *
from collections import namedtuple

learnedDataNames = ['a', 'ba', 'ka', 'pa']
learnedDataFILES = ['data/moji_' + fileName + '.dat' for fileName in learnedDataNames ]
testDataFiles = ['data/moji_test.dat']


data = readDataFrom(learnedDataFILES, columnNames=learnedDataNames)
testData = readDataFrom(testDataFiles, columnNames=['test'])

results = {
    'euclidean': euclideanMethod(data, testData),
    'similarity': similarityMethod(data, testData),
    'weightDistance': euclideanMethod(data, testData, weight=pd.Series(range(256))),
    'pca': pcaMethod(data, testData)
}

correctClasses = testData['test'].loc[:, 256].transform(lambda x: learnedDataNames[x-1])
# 'euclidean': results['euclidean'] == correctClasses
errata = { key: value == correctClasses for key, value in results.items() }

# 'euclidean': errata['euclidean'].mean()
recognitionRates = { key: value.mean() for key, value in errata.items() }
print(recognitionRates)
