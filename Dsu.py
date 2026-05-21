class DSU:
    n = 0
    parents = [0]

    def __init__(self, n):
        self.n =n
        for i in range(1, n+1):
            self.parents.append(i)

    def find(self, x):
        while(x != self.parents[x]):
            self.parents[x] = self.parents[self.parents[x]]
            x = self.parents[x]
        return x
    
    def uni(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if(x == y):
            return 0
        
        self.parents[x] = y
        return 1
    
    

    



