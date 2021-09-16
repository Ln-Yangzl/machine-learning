from KNNclassifier import KNNclassifier
import numpy as np
from dataReader import loadMNIST

DATA_PATH = 'data/'
BATCH_SIZE = 10

knn = KNNclassifier(neighbors=10, compute=lambda x,y:(x-y)*(x-y))

xTrain, xTest, yTrain, yTest = loadMNIST(DATA_PATH)
knn.fit(xTrain[:1000], yTrain[:1000])
index = 0
record = 0.0
while index + BATCH_SIZE <= len(xTest):
    end = index + BATCH_SIZE 
    print('batch: ', index, '----', end)
    yPred = knn.predict(xTest[index:end])
    current = np.mean(yPred == yTest[index:end])
    record += current
    print('acc: {:.5f}'.format(current))
    index += BATCH_SIZE

print('Test complished !')
print('Acc: {:.5f}'.format(record/(len(xTest)/BATCH_SIZE)))

# yPred = knn.predict(xTest)