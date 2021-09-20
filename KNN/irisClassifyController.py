from sklearn import datasets
from sklearn.model_selection import train_test_split
from KNNclassifier import KNNclassifier
import numpy as np

testNums = 10

iris = datasets.load_iris()
x = iris.data
y = iris.target


manhattanDistance = lambda x,y:abs(x-y)
euclideanDistance = lambda x,y:(x-y)*(x-y)
knn = KNNclassifier(compute = euclideanDistance)

# res = knn.predictSingle([6.9,3.1,5.1,2.3])

bestNeighbors = 0
bestAcc = 0.0
# xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.25)
for i in range(1, 21):
    knn.neighbors = i
    # print(yPred)
    print("neighbors:", i)
    # print("accuracy:")
    record = 0
    for k in range(testNums):
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.4)
        knn.fit(xTrain, yTrain)
        yPred = knn.predict(xTest)
        # record += "{:.5f}".format(np.mean(yPred == yTest))
        current = np.mean(yPred == yTest)
        record += current
        # print("{:.5f}".format(current),end=' ')
    acc = record/testNums
    print("average: ", "{:.5f}".format(acc))
    if acc > bestAcc:
        bestAcc = acc
        bestNeighbors = i

print("bestAcc: {:.5f}".format(bestAcc), "bestNeighbors:", bestNeighbors)

