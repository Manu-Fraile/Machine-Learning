# -*- coding: utf-8 -*-

import random
import numpy as np
import math as m
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# -------- CODE STARTS HERE -------- #



# set kernel functions (we should readjust standard values)
def linearKernel(x, y):
    return np.dot(x, y) + 1

def polyKernel(x, y, p=3):
    return np.power((np.dot(x, y) + 1), p)

def radialKernel(x, y, sigma=2):
    diff = np.subtract(x, y)
    return m.exp((-np.dot(diff, diff)) / (2 * sigma * sigma))



# create the dataset
def dataset():
    
    classA = np.concatenate(
            (np.random.randn(10, 2)*0.2 + [1.5, 0.5],
             np.random.randn(10, 2)*0.2 + [-1.5, 0.5]))
    classB = np.random.randn(20, 2)*0.2 + [0.0, -0.5]
    
    inputs = np.concatenate((classA, classB))
    targets = np.concatenate(
            (np.ones(classA.shape[0]),
            -np.ones(classB.shape[0])))
    
    N = inputs.shape[0]    #number of rows (samples)
    
    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]
    
    return classA, classB, targets, inputs, permute     #the last three have 40 datapoints. The other 2 20.



#zerofun function   
def zerofun(alpha):
    return sum([alpha[i] * t[i] for i in range(N)])



#indicator function
#def indicator(a, x, t, kernel):
#    totsum = 0
#    for value in nonzero:
#        totsum += value[0] * value[2] * kernel([a, x], value[1])
#    return totsum - t 
    
def indicator(sv, alphas, inputs, targets, b):
    sm = 0
    for i in range(len(alphas)):
        sm += alphas[i]*targets[i]*kernel(sv,inputs[i])
    sm -= b
    return sm



#objective function and what it needs
def p_matrix(inputs, targets, N):
    P = []
    for i in range(N):
        A = []
        for j in range(N):
            k = kernel(inputs[i], inputs[j])
            A.append(targets[i]*targets[j]*k)
        P.append(np.array(A))

    return np.array(P)

def objective(alpha_vector):
    
    alpha_product = -sum(alpha_vector)
    for i in range(N):
        for j in range(N):
            procedure = 0.5 * alpha_vector[i] * alpha_vector[j] * p_matrix[i][j]
            alpha_product += procedure

    return alpha_product
    


#b value
def b_value(alphas, inputs, targets, C):
    si = 0
    for i in range(len(alphas)):
        if alphas[i] < C:
            si = i
            break
    ans = 0
    for i in range(len(inputs)):
        ans += alphas[i]*targets[i]*kernel(inputs[si], inputs[i])
    return ans - targets[si]



def plotting(classA, classB):
    
    plt.plot([p[0] for p in classA],
             [p[1] for p in classA],
             'b.')
    plt.plot([p[0] for p in classB],
             [p[1] for p in classB],
             'r.')
    
    plt.axis('equal')     #Force same scale on both axes
    #plt.savefig('svmplot.pdf')     #Save a copy in a file
    plt.show()     #Show the plot on the screen


if __name__=="__main__":
    
    # kernel selection
    kernel = linearKernel
    
    # dataset creation
    classA, classB, targets, inputs, permute = dataset()
    
    N = inputs.shape[0]
    start = np.zeros(N)
    C = None
    
    bounds=[(0, None) for b in range(N)]
    constraint={'type':'eq', 'fun': zerofun}
       
    p_matrix = p_matrix(inputs, targets, N)
    threshold = m.pow(10, -5)
    
    print(len(permute))
    print(classB.shape)
    
    #plt.scatter(inputs[:,0],inputs[:,1])
    
    #our objective is to call this function
    ret = minimize(objective, start, bounds=bounds, constraints=constraint)
    
    print(ret['x'])
    
    plotting(classA, classB)
    
    
    
# -------- CODE FINISHES HERE -------- #





# -------- NOTES START HERE -------- #

#ret = minimize(objective, start, bounds=B, constraints= XC)

#alpha = ret['x']


def objective(a_vector):
    
    return 'scalar'

n = 100                  #number of training samples
start = np.zeros(n)      #the initial guess of a_vector

#B = ([1,2],[0.4,0.5],[65,71],...,[n1,n2])
#bounds = [(0,C) for b in range(n)]

#a = [expr for x in seq]     creates a list with the same length as 'seq'


def zerofun():
    
    return 'algo'

b = 0

for i in range(n):
    
    #param = a[i]*t[i]*K(s,x[i])-t
    b = 1


def indicator(a, x, t):
    
    return('algo')
    
    
def dataset():
    
    classA = np.concatenate(
            (np.random.randn(10, 2)*0.2 + [1.5, 0.5],
             np.random.randn(10, 2)*0.2 + [-1.5, 0.5]))
    classB = np.random.randn(20, 2)*0.2 + [0.0, -0.5]
    
    inputs = np.concatenate((classA, classB))
    targets = np.concatenate(
            (np.ones(classA.shape[0]),
            -np.ones(classB.shape[0])))
    
    N = inputs.shape[0]    #number of rows (samples)
    
    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]
    
    
def plotting(classA, classB):
    
    plt.plot([p[0] for p in classA],
             [p[1] for p in classA],
             'b.')
    plt.plot([p[0] for p in classB],
             [p[1] for p in classB],
             'r.')
    
    plt.axis('equal')     #Force same scale on both axes
    #plt.savefig('svmplot.pdf')     #Save a copy in a file
    plt.show()     #Show the plot on the screen
    
    
def plot_DB():
    
    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)
    
    grid = np.array([[indicator(x, y)
                      for x in xgrid]
                      for y in ygrid])
    
    plt.contour(xgrid, ygrid, grid,
                (-1.0, 0.0, 1.0),
                colors=('red','black','blue'),
                linewidths=(1, 3, 1))
    
   
# -------- NOTES FINISH HERE -------- #
    