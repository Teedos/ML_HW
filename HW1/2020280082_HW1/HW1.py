# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file

def parse(file, n_features):
    data = load_svmlight_file(file, n_features)
    x = data[0]
    y = data[1]
    for i in range(len(y)):
        if (y[i] == -1):
            y[i] = 0
    return x.toarray().T,y

def sigmoid(x, w):  
    arg = -w.T.dot(x)
    return 1/ (1 + np.exp(arg))

    
def irls(X, w, y, max_iter):
    n_iter = 0;
    while (n_iter < max_iter):
        n_iter = n_iter +1
        miu = sigmoid(X,w)
        R = np.array(miu*(1-miu))
        R = np.diag(R)
        print("R shape: ", R.shape)
        temp = np.linalg.pinv(((X*R).dot(X.T))).dot( (X*R).dot(X.T)-X.dot(y - miu) ) 
        w = w - temp
    return w

n_features = 123
X, Y = parse("a9a.t", n_features)
print("X shape",X.shape)
w = np.zeros(n_features)

#print(get_R(X,w))
irls_output = irls(X, w, Y, 10)
print(irls_output)
