# from tools import kVector

# v = kVector(3, lambda x:x[0])

# print(v.data)
# v.push((2,3))
# v.push((1,5))
# v.push((2,6))
# v.push((4,2))

# print(v.data)
# for i in range(1,10):
#     print(i)

import matplotlib.pyplot as plt

with open('data/train-images.idx3-ubyte', 'rb') as file:
    file.read(4)
    size = int(file.read(4).hex(), 16)
    print('size: ', size)
    rows = int(file.read(4).hex(), 16)
    columns = int(file.read(4).hex(), 16)

    while True:
        img = []
        for i in range(28):
            row = []
            for j in range(28):
                row.append(int(file.read(1).hex(), 16))
            img.append(row)
        print('img show')
        plt.imshow(img, cmap='gray')
        plt.show()
        
