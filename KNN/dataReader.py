res = []

with open("test.txt", 'rb') as f:
    # data = int(f.read(4).hex(), 16)
    data = f.read(4)
    res.append(data)
    print(data)
    data = f.read(4)
    print(type(data))
    print(len(data))
    res.append(data)
    print(data)
f.close()

print(res)