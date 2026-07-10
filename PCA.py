import numpy as np


class PCA:
    def __init__(self, dimensions = 2):
        self.dimensions = dimensions

    def cov(self, data):
        return (data.T @ data)/(data.shape[0] - 1)

    def fit(self, data):

        #normalize data
        data = data.copy()
        #center data around the mean and scale so that variance is of unit length
        data = (data - np.mean(data, axis = 0)) / np.std(data, axis = 0)


        #find covariance matrix
        comat = self.cov(data)

        #find eigenvectors and values
        eVals, eVecs = np.linalg.eigh(comat)

        pairs = []

        for i in range(comat.shape[0]):
            #row -> columns

            pairs.append((eVals.T[i], eVecs.T[i]))
        
        pairs.sort(key = lambda x : x[0], reverse = True)

        eVecs = []
        eVals = []

        for i in pairs:
            eVecs.append(i[1])    
            eVals.append(i[0])

        self.components = eVecs[:self.dimensions]
        self.explained_variance = eVals[:self.dimensions]

        return eVals[:self.dimensions], np.array(eVecs[:self.dimensions])
    
    def transform(self, data):
        X = (data - self.mean_) / self.std_
        return X @ self.components_

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

        

            
        

