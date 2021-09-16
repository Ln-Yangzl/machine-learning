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

# import matplotlib.pyplot as plt

# with open('data/train-images.idx3-ubyte', 'rb') as file:
#     file.read(4)
#     size = int(file.read(4).hex(), 16)
#     print('size: ', size)
#     rows = int(file.read(4).hex(), 16)
#     columns = int(file.read(4).hex(), 16)

#     while True:
#         img = []
#         for i in range(28):
#             row = []
#             for j in range(28):
#                 row.append(int(file.read(1).hex(), 16))
#             img.append(row)
#         print('img show')
#         plt.imshow(img, cmap='gray')
#         plt.show()
        
import math
import datetime
import multiprocessing as mp


def train_on_parameter(name, param):
    result = 0
    for num in param:
        result += math.sqrt(num * math.tanh(num) / math.log2(num) / math.log10(num))
    return {name: result}


if __name__ == '__main__':

    start_t = datetime.datetime.now()

    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(num_cores)
    param_dict = {'task1': list(range(10, 30000)),
                  'task2': list(range(30000, 60000)),
                  'task3': list(range(60000, 90000)),
                  'task4': list(range(90000, 120000)),
                  'task5': list(range(120000, 150000)),
                  'task6': list(range(150000, 180000)),
                  'task7': list(range(180000, 210000)),
                  'task8': list(range(210000, 240000))}
    results = [pool.apply_async(train_on_parameter, args=(name, param)) for name, param in param_dict.items()]
    print(results)
    results = [p.get() for p in results]
    print(results)

    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("多进程计算 共消耗: " + "{:.2f}".format(elapsed_sec) + " 秒")