from tools import kVector

v = kVector(3, lambda x:x[0])

print(v.data)
v.push((2,3))
v.push((1,5))
v.push((2,6))
v.push((4,2))

print(v.data)
for i in range(1,10):
    print(i)