from tools import kVector


class KNNclassifier:

    def __init__(self, neighbors = 10, compute = lambda x,y:(x-y)*(x-y)):
        self.neighbors = neighbors
        self.compute = compute
        

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, xTest):
        yPredict = []
        for x in xTest:
            yPredict.append(self.predictSingle(x))
        return yPredict

    def predictSingle(self, x):
        # vector 用于存储最小的k个数据，vector会自动判断该数据是该被添加或舍弃
        vector = kVector(self.neighbors, lambda x:x[0])
        for i in range(len(self.x)):
            distence = 0
            for k in range(len(x)):
                distence += self.compute(x[k], self.x[i][k])
            # 将当前数据添加到vector中
            vector.push((distence, self.y[i]))
        # 统计k个中出现最多的类别
        cont = [0]*len(self.y)
        max = 0
        maxY = 0
        for i in vector.data:
            currentY = i[-1]
            cont[currentY] += 1
            if cont[currentY] > max:
                max = cont[currentY]
                maxY = currentY

        return maxY



