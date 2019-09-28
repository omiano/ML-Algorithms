import numpy as np
import random
import math

def perceptron(trainingData, numPasses):
    w = np.zeros(819)
    for i in range(numPasses):
        for line in trainingData:
            x = np.array(line[:819])
            y = line[819]
            if y == 2:
                y = -1
            if y*np.dot(w,x) <= 0:
                w = w + y*x
    return w

def kernelizedPerceptron(trainingData, numPasses, K):
    M = []
    for i in range(numPasses):
        sum = 0
        t = 0
        for line in trainingData:
            x_t = np.array(line[:819])
            y_t = line[819]
            if y_t == 2:
                y_t = -1
            sum = 0
            for i in M:
                x_i = np.array(trainingData[i][:819])
                y_i =  trainingData[i][819]
                if y_i == 2:
                    y_i = -1
                sum = sum + y_i*K(x_i, x_t)
            if y_t*sum <= 0:
                M.append(t)
            t = t + 1
    return M
