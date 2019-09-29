from knn import kNN
from id3 import id3, pruneTree
from perceptron import perceptron, kernelizedPerceptron
from boosting import boost, strongLearner
import numpy as np
import random
import math

# kNN Example Use

# get training data and vector to test
knnTrainingData = open('data/knntrain.txt', 'r').readlines()

# get projection matrix
projection = open('data/projection.txt', 'r').readlines()
projMatrix = []
for line in projection:
    projMatrix.append(np.fromstring(line, dtype=float, sep=' '))
projMatrix = np.array(projMatrix)

# get training matrix
knnTrainingMatrix = []
for line in knnTrainingData:
    knnTrainingMatrix.append(np.fromstring(line, dtype=int, sep=' '))
knnTrainingMatrix = np.array(knnTrainingMatrix)

# get labels
labels = []
for row in knnTrainingMatrix:
    labels.append(row[-1])
knnTrainingMatrix = np.delete(knnTrainingMatrix, 784, 1)

# projected training matrix
productMatrix = knnTrainingMatrix.dot(projMatrix)

# get test matrix
knnTestData = open('data/knntest.txt', 'r').readlines()
knnTestMatrix = []
for line in knnTestData:
    knnTestMatrix.append(np.fromstring(line, dtype=int, sep=' '))
knnTestMatrix = np.array(knnTestMatrix)

# normal kNN test error
numErrors = 0
for row in knnTestMatrix:
    vector = row[:-1]
    predictedLabel = kNN(vector, knnTrainingMatrix, labels, 15)
    if predictedLabel != row[-1]:
        numErrors = numErrors + 1
print(float(numErrors)/len(knnTestMatrix))

# projected kNN test error
numErrors = 0
for row in knnTestMatrix:
    vector = row[:-1]
    projVector = vector.dot(projMatrix)
    predictedLabel = kNN(projVector, productMatrix, labels, 15)
    if predictedLabel != row[-1]:
        numErrors = numErrors + 1
print(float(numErrors)/len(knnTestMatrix))


# ID3 Example Use

# get training data
id3TrainingData = open('data/id3train.txt', 'r').readlines()
id3TrainingMatrix = []
for line in id3TrainingData:
    id3TrainingMatrix.append(np.fromstring(line, dtype=float, sep=' '))
id3TrainingMatrix = np.array(id3TrainingMatrix)

# get test data
id3TestData = open('data/id3test.txt', 'r').readlines()
id3TestMatrix = []
for line in id3TestData:
    id3TestMatrix.append(np.fromstring(line, dtype=float, sep=' '))
id3TestMatrix = np.array(id3TestMatrix)

# ID3 test error without pruning
rootNode = id3(id3TrainingMatrix)
numErrors = 0
for row in id3TestMatrix:
    currNode = rootNode
    while isinstance(currNode.data, list):
        featureIndex = currNode.data[1]
        featureVal = currNode.data[0]
        if row[featureIndex] <= featureVal:
            currNode = currNode.yesPtr
        else:
            currNode = currNode.noPtr
    if currNode.data != row[-1]:
        numErrors = numErrors + 1
print(float(numErrors)/len(id3TestMatrix))

# ID3 test error with pruning
pruneTree(rootNode)
numErrors = 0
for row in id3TestMatrix:
    currNode = rootNode
    while isinstance(currNode.data, list):
        featureIndex = currNode.data[1]
        featureVal = currNode.data[0]
        if row[featureIndex] <= featureVal:
            currNode = currNode.yesPtr
        else:
            currNode = currNode.noPtr
    if currNode.data != row[-1]:
        numErrors = numErrors + 1
print(float(numErrors)/len(id3TestMatrix))


# Perceptron Example Use

# get training data
perTrainingData = open('data/pertrain.txt', 'r').readlines()
perTrainingMatrix = []
for line in perTrainingData:
    perTrainingMatrix.append(np.fromstring(line, dtype=float, sep=' '))
perTrainingMatrix = np.array(perTrainingMatrix)

# get test data
perTestData = open('data/pertest.txt', 'r').readlines()
perTestMatrix = []
for line in perTestData:
    perTestMatrix.append(np.fromstring(line, dtype=float, sep=' '))
perTestMatrix = np.array(perTestMatrix)

# kernel function
def K(x, z):
    val = np.linalg.norm(x-z, 1)
    val = val/20
    return math.exp(-val)

# Linear Perceptron Test Error
numErrors = 0
w = perceptron(perTrainingMatrix, 2)
for row in perTestMatrix:
    label = 0
    if np.dot(w, row[:-1]) < 0:
        label = 2
    elif np.dot(w, row[:-1]) > 0:
        label = 1
    else:
        label = random.randint(1, 2)
    if label != row[-1]:
        numErrors = numErrors + 1
print(float(numErrors)/len(perTestMatrix))

# Kernelized Perceptron Test Error
numErrors = 0
M = kernelizedPerceptron(perTrainingMatrix, 4, K)
for row in perTestMatrix:
    currsum = 0
    for i in M:
        y_i = perTrainingMatrix[i][-1]
        if y_i == 2:
            y_i = -1
        x_i = perTrainingMatrix[i][:-1]
        x = np.array(row[:-1])
        currsum = currsum + y_i*K(x_i, x)
    label = 0
    if currsum < 0:
        label = 2
    elif currsum > 0:
        label = 1
    else:
        label = random.randint(1, 2)
    if label != row[-1]:
        numErrors = numErrors + 1
print(float(numErrors)/len(perTestMatrix))


# Boosting Example Use

# get training data
boostTrainingData = open('data/boosttrain.txt', 'r').readlines()
boostTrainingMatrix = []
for line in boostTrainingData:
    boostTrainingMatrix.append(np.fromstring(line, dtype=float, sep=' '))
boostTrainingMatrix = np.array(boostTrainingMatrix)

# get test data
boostTestData = open('data/boosttest.txt', 'r').readlines()
boostTestMatrix = []
for line in boostTestData:
    boostTestMatrix.append(np.fromstring(line, dtype=float, sep=' '))
boostTestMatrix = np.array(boostTestMatrix)

# get an array of weak learners
f = boost(boostTrainingMatrix, 4)

# get test error
numErrors = 0
for row in boostTestMatrix:
    label = strongLearner(f, row)
    if label != row[-1]:
        numErrors = numErrors + 1
print(float(numErrors)/len(boostTestMatrix))
