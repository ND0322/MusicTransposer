import numpy as np

class PCA:
    def __init__(self, dimensions = 2):
        self.dimensions = 2

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
        

