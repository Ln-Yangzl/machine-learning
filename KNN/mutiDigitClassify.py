import multiprocessing as mp
from KNNclassifier import KNNclassifier
import numpy as np
from dataReader import loadMNIST
import datetime

DATA_PATH = 'data/'
BATCH_SIZE = 10
PROCESSING_SIZE = 200
PROCESS_NUM = 8

def classify(name, x, y, knn):
    index = 0
    record = 0.0
    while index + BATCH_SIZE <= len(x):
        end = index + BATCH_SIZE
        print(name + ' batch: ', index, '----', end)
        yPred = knn.predict(x[index:end])
        current = np.mean(yPred == y[index:end])
        record += current
        print(name + 'acc: {:.5f}'.format(current))
        index += BATCH_SIZE
    return record

def computeFun(x, y):
    return (x-y)*(x-y)

if __name__ == '__main__':

    xTrain, xTest, yTrain, yTest = loadMNIST(DATA_PATH)
    knn = []
    for i in range(PROCESS_NUM):
        temp = KNNclassifier(neighbors=10, compute=computeFun)
        temp.fit(xTrain[:1000], yTrain[:1000])
        knn.append(temp)


    num_cores = int(mp.cpu_count())
    print('starting ', PROCESS_NUM, ' processes')
    pool = mp.Pool(PROCESS_NUM)

    param_pack = []
    index = 0
    for i in range(PROCESS_NUM):
        start = index
        end = index + PROCESSING_SIZE
        param_pack.append(
            ('task' + str(i),
            xTest[start:end],
            yTest[start:end],
            knn[i])
        )
        index += PROCESSING_SIZE

    start_t = datetime.datetime.now()

    results = [pool.apply_async(classify, args=param) for param in param_pack]  
    count = 0.0
    # print(results)
    # results = [p.get() for p in results]
    # print(results)
    for p in results:
        count += p.get()
    
    print('Test finished !')
    print('Acc: {:.5f}'.format(count / (PROCESSING_SIZE / BATCH_SIZE * PROCESS_NUM)))
    
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print('total time: ' + "{:.2f}".format(elapsed_sec) + 's')
