from knn import kNN
import numpy as np

# kNN Example Use

# get training data and vector to test
trainingData = open('knntrain.txt', "r").readlines()

# get projection matrix
projection = open('projection.txt', "r").readlines()
projMatrix = []
for line in projection:
    projMatrix.append(np.fromstring(line, dtype=float, sep=' '))
projMatrix = np.array(projMatrix)

# get training matrix
trainingMatrix = []
for line in trainingData:
    trainingMatrix.append(np.fromstring(line, dtype=int, sep=' '))
trainingMatrix = np.array(trainingMatrix)

# get labels
labels = []
for row in trainingMatrix:
    labels.append(row[-1])
trainingMatrix = np.delete(trainingMatrix, 784, 1)

# projected training matrix
productMatrix = trainingMatrix.dot(projMatrix)

# get test matrix
testData = open('knntest.txt', "r").readlines()
testMatrix = []
for line in testData:
    testMatrix.append(np.fromstring(line, dtype=int, sep=' '))
testMatrix = np.array(testMatrix)

# normal kNN test error
numErrors = 0
for row in testMatrix:
    vector = row[:-1]
    predictedLabel = kNN(vector, trainingMatrix, labels, 15)
    if predictedLabel != row[-1]:
        numErrors = numErrors + 1
print(float(numErrors)/len(testMatrix))

# projected kNN test error
numErrors = 0
for row in testMatrix:
    vector = row[:-1]
    projVector = vector.dot(projMatrix)
    predictedLabel = kNN(projVector, productMatrix, labels, 15)
    if predictedLabel != row[-1]:
        numErrors = numErrors + 1
print(float(numErrors)/len(testMatrix))

