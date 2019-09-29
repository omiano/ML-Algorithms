from scipy import stats
from collections import deque
import numpy as np
import operator
import copy

class Node:
    yesPtr = 0
    noPtr = 0

    def __init__(self, data):
        self.data = data
        count = 0
        for row in data:
            if row[-1] == 1:
                count = count + 1
        if count > (len(data) - count):
            self.majorityLabel = 1
        else:
            self.majorityLabel = 0

def id3(trainingData):
    rootNode = Node(trainingData)

    q = deque()
    q.append(rootNode)
    while len(q) > 0:
        currNode = q.popleft()
        # check if current node is pure
        isPure = True
        firstLabel = currNode.data[0][-1]
        for row in currNode.data:
            if row[-1] != firstLabel:
                isPure = False
                break
        if isPure:
            currNode.data = firstLabel
            continue
        # get feature split with minimum entropy, aka best decision
        entropy = getMinEntropy(currNode.data)
        featureVal = entropy[0]
        index = entropy[1]
        # create child nodes after split
        yesArr = []
        noArr = []
        for row in currNode.data:
            if row[index] <= featureVal:
                yesArr.append(row)
            else:
                noArr.append(row)
        currNode.data = [featureVal, index]
        currNode.yesPtr = Node(yesArr)
        currNode.noPtr = Node(noArr)
        # add child nodes to queue
        q.append(currNode.yesPtr)
        q.append(currNode.noPtr)
    return rootNode

# get decision with overall minimum entropy
def getMinEntropy(trainingData):
    minEntropy = 100
    feature = 0
    val = 0
    for i in range(len(trainingData[0])):
        entropy = getFeatureEntropy(trainingData, i)
        if entropy[0] < minEntropy:
            minEntropy = entropy[0]
            val = entropy[1]
            feature = i
    return (val, feature)

# gets split with minimum entropy per feature
def getFeatureEntropy(trainingData, feature):
    # get number of occurrences for each value of feature
    attr = getAttrVals(trainingData, feature)
    attrVals = attr[0]
    attrVals.sort(key=operator.itemgetter(0))
    attrSize = attr[1]

    # get midpoints of feature values
    midPoints = []
    for i in range(len(attrVals) - 1):
        midPoints.append((attrVals[i][0] + attrVals[i+1][0])/2)
    # array of conditional probabilities
    probs = getProbability(midPoints, attrVals, attrSize)
    probabilities = probs[0]
    jointProbabilities1 = probs[1]
    jointProbabilities2 = probs[2]
    condProbabilities1 = []
    condProbabilities2 = []
    for i in range(len(probabilities)):
        condProbabilities1.append(jointProbabilities1[i]/probabilities[i])
        condProbabilities2.append(jointProbabilities2[i]/(1 - probabilities[i]))

    minEntropy = 100
    index = 0
    # find split with lowest entropy
    for i in range(len(condProbabilities1)):
        entropy1 = stats.entropy([condProbabilities1[i], 1 - condProbabilities1[i]])
        entropy2 = stats.entropy([condProbabilities2[i], 1 - condProbabilities2[i]])
        condEntropy = probabilities[i]*entropy1 + (1 - probabilities[i])*entropy2
        if condEntropy < minEntropy:
            minEntropy = condEntropy
            index = i
    # return the entropy
    if len(midPoints) == 0:
        return [minEntropy, 0]
    return [minEntropy, midPoints[index]]

# returns array of tuples of the form (feature_val, num_occurrences, prob_num)
# feature_val is a value of the feature vector , num_occurrences is how many 
# times that value appears in the feature vector, prob_num is how many components
# of the feature vector with that value have a label 1
def getAttrVals(trainingData, feature):
    attrVals = []
    attrSize = 0
    for row in trainingData:
        valIndex = -1
        i = 0
        for val in attrVals:
            if val[0] == row[feature]:
                valIndex = i
                break
            i = i + 1
        if valIndex != -1:
            numOccurrences = operator.getitem(attrVals, valIndex)[1]
            probNum = operator.getitem(attrVals, valIndex)[2]
            if row[-1] == 1:
                operator.setitem(attrVals, valIndex, (row[feature], numOccurrences + 1, probNum + 1))
            else:
                operator.setitem(attrVals, valIndex, (row[feature], numOccurrences + 1, probNum))
        else:
            if row[-1] == 1:
                attrVals.append((row[feature], 1, 1))
            else:
                attrVals.append((row[feature], 1, 0))
        attrSize = attrSize + 1
    return (attrVals, attrSize)

# calculates probabilities
def getProbability(midPoints, attrVals, attrSize):
    probabilities = []
    jointProbabilities1 = []
    jointProbabilities2 = []
    for val in midPoints:
        prob = 0
        jointProb1 = 0
        jointProb2 = 0
        for pair in attrVals:
            if pair[0] <= val:
                prob = prob + pair[1]
                jointProb1 = jointProb1 + pair[2]
            else:
                jointProb2 = jointProb2 + pair[2]
        probabilities.append(prob/float(attrSize))
        jointProbabilities1.append(jointProb1/float(attrSize))
        jointProbabilities2.append(jointProb2/float(attrSize))

    return (probabilities, jointProbabilities1, jointProbabilities2)

# prune the tree
def pruneTree(rootNode):
    q = deque()
    q.append(rootNode)
    minError = 101
    nodeToPrune = rootNode
    # try pruning at every node
    while len(q) > 0:
        currNode = q.popleft()
        if isinstance(currNode.data, list) == False:
            continue
        error = testNewTree(rootNode, currNode)
        if error < minError:
            minError = error
            nodeToPrune = currNode
        q.append(currNode.yesPtr)
        q.append(currNode.noPtr)
    # prune at node that had tree with lowest error
    nodeToPrune.data = nodeToPrune.majorityLabel
    nodeToPrune.yesPtr = 0
    nodeToPrune.noPtr = 0

# get validation error when pruning at nodeToPrune
def testNewTree(root, nodeToPrune):
    # get validation data
    validationData = open('data/id3validation.txt', "r").readlines()
    validationMatrix = []
    for line in validationData:
        validationMatrix.append(np.fromstring(line, dtype=float, sep=' '))
    validationMatrix = np.array(validationMatrix)

    # traverse through tree
    numErrors = 0
    numRows = 0
    for row in validationMatrix:
        currNode = root
        while isinstance(currNode.data, list):
            # return majority label at nodeToPrune
            if currNode == nodeToPrune:
                currNode = Node([])
                currNode.data = nodeToPrune.majorityLabel
                break
            featureIndex = currNode.data[1]
            featureVal = currNode.data[0]
            if row[featureIndex] <= featureVal:
                currNode = currNode.yesPtr
            else:
                currNode = currNode.noPtr
        if currNode.data != row[-1]:
            numErrors = numErrors + 1
        numRows = numRows + 1
    return float(numErrors)/numRows
