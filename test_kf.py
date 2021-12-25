'''
Sidney Liu
Unit testing for Kalman Filter
CS321
December 22, 2021
'''

from Classifier import KF
import numpy as np

class test_kf:
    def test_predict(self, obj, mew):
        '''
        Parameters:
        -----------
        mew: external motion scalar. (dotted with b)
        '''
        obj.predict(mew)
    
    def test_update(self, obj, z):
        '''
        Parameters:
        -----------
        z: matrix of measurements. shape=(n, n[0])
        '''
        obj.update(z)
    
if __name__=="__main__":
    dt = 1.0/60
    F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
    H = np.array([1, 0, 0]).reshape(1, 3)
    Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
    R = np.array([0.5]).reshape(1, 1)
    x = np.linspace(-10, 10, 100)
    measurements = - (x**2 + 2*x - 2) + np.random.normal(0, 2, 100)

    tkf = test_kf()
    obj = KF(F = F, H = H, Q = Q, R = R)

    print(tkf.test_predict(obj, 0))
    print(tkf.test_update(obj, 0))