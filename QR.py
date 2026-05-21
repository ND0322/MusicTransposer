import numpy as np


class EigenDecomp():
    def power_iteration(A, tol = 0.0001):
        v = np.random.normal(size = A.shape[1])
        

        v = v/np.linalg.norm(v)
        steps = [v]

        last = np.empty(shape = A.shape[1])
        while(True):
            last = v.copy()
            v = A @ v

            v = v / np.linalg.norm(v)

            steps.append(v)

            if np.allclose(v, last, atol = tol) or np.allclose(v, -last, atol = tol):
                break

        eigValue = v @ (A @ v)
        return eigValue, v, steps
    

    def QR(A, tol = 0.0001):
        Q,R = np.linalg.qr(A)

        last = np.empty(shape = Q.shape)

        for i in range(500):
            last = Q.copy()
            X = Q @ R

            Q,R = np.linalg.qr(X)

            if(np.allclose(X, np.triu(X)), atol = tol):
                break
        return Q

