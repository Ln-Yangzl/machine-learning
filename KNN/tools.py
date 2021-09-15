
class kVector:

    def __init__(self, size, key = lambda x:x):
        self.size = size
        self.data = []
        self.key = key

    def push(self, item):
        if len(self.data) < self.size:
            self.data.append(item)
        elif self.key(self.data[-1]) > self.key(item):
            self.data[-1] = item
        self.data.sort(key=self.key) 

    def getData(self):
        return self.data   