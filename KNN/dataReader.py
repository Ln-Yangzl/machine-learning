import numpy as np
from io import BufferedReader


def loadMNIST(path:str = ''):

    with open(path + 'train-images.idx3-ubyte', 'rb') as f:
        print('reading images from train-images.idx3-ubyte...')
        xTrain = readImages(f)
        print('complete')
        # print(len(xTrain))
        # print(xTrain)
    f.close()
    with open(path + 'train-labels.idx1-ubyte', 'rb') as f:
        print('reading labes from train-labels.idx1-ubyte...')
        yTrain = readLables(f)
        print('complete')
        # print(len(yTrain))
        # print(yTrain)
    f.close()
    with open(path + 't10k-images.idx3-ubyte', 'rb') as f:
        print('reading images from t10k-images.idx3-ubyte...')
        xTest = readImages(f)
        print('complete')
    f.close()
    with open(path + 't10k-labels.idx1-ubyte', 'rb') as f:
        print('reading lables from t10k-labels.idx1-ubyte...')
        yTest = readLables(f)
        print('complete')
    f.close()
    return xTrain, xTest, yTrain, yTest


def readImages(file:BufferedReader):
    file.read(4)
    size = int(file.read(4).hex(), 16)
    print('size: ', size)
    rows = int(file.read(4).hex(), 16)
    columns = int(file.read(4).hex(), 16)
    imagePixels = rows*columns
    
    res = []
    for i in range(size):
        image = []
        for k in range(imagePixels):
            image.append(int(file.read(1).hex(), 16))
        res.append(image)
    return np.asarray(res)

def readLables(file:BufferedReader):
    file.read(4)
    size = int(file.read(4).hex(), 16)
    print('size: ', size)
    res = []
    for i in range(size):
        res.append(int(file.read(1).hex(), 16))
    return np.asarray(res)


# loadMNIST('data/')
