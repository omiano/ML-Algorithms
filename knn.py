from scipy.spatial import distance
from scipy import stats
import operator
import numpy as np
import time

def kNN(vector, data, labels, k):
    distArr = []
    rowNum = 0
    for row in data:
        dist = distance.euclidean(vector, row)
        distArr.append((labels[rowNum], dist))
        rowNum = rowNum + 1
        
    distArr.sort(key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        neighbors.append(distArr[i][0])
    mode = stats.mode(neighbors)
    return mode[0][0]
