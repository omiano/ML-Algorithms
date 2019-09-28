from knn import kNN
from id3 import id3, pruneTree
import numpy as np

# kNN Example Use

# get training data and vector to test
knnTrainingData = open('knntrain.txt', 'r').readlines()

# get projection matrix
projection = open('projection.txt', 'r').readlines()
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
knnTestData = open('knntest.txt', 'r').readlines()
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
id3TrainingData = open('id3train.txt', 'r').readlines()
id3TrainingMatrix = []
for line in id3TrainingData:
    id3TrainingMatrix.append(np.fromstring(line, dtype=float, sep=' '))
id3TrainingMatrix = np.array(id3TrainingMatrix)

# get test data
id3TestData = open('id3test.txt', 'r').readlines()
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

