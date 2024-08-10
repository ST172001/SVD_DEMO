from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
import time
def get_eigen_values_and_vectors(A:np.array):
    eigen_values,eigen_vectors=np.linalg.eig(A)
    return eigen_values,eigen_vectors

def compute_svd(A:np.array):
    st=time.time()
    eig_values1,right_sing_vectors=get_eigen_values_and_vectors(np.dot(np.transpose(A),A))
    return eig_values1,right_sing_vectors

def get_approximate_image(X,num_singular,eigen_values,right_vs):
    #construct a new matrix
    Y=np.zeros((X.shape[0],X.shape[1]),dtype=np.complex128)
    for i,eig_value in enumerate(eigen_values):
        if i>=num_singular:
            break
        right_vi=right_vs[:,i].reshape(X.shape[1],1)
        number=np.sqrt(eig_value)
        if number>0:
            left_ui=(np.dot(X,right_vi/number)).reshape(X.shape[0],1)
            Y+=np.sqrt(eig_value)*np.dot(left_ui,np.transpose(right_vi))
    return Y      