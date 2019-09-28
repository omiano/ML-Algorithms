import numpy as np
import random

# main boosting function, outputs an array of the chosen weak learners and 
# their corresponding alphas at each round
def boost(trainingData, numRounds):
    # initialize distribution uniformly
    D = [1.0/len(trainingData)]*len(trainingData)
    # array of weak learners
    chosenLearners = []

    for t in range(numRounds):
        # get best learner and its error for this round
        learnerAndError = selectBestLearner(trainingData, D)
        bestLearner = learnerAndError[0]
        epsilon = learnerAndError[1]
        alpha = np.log((1 - epsilon)/epsilon)/2
        #update D
        for i in range(len(trainingData)):
            h_x = h(bestLearner[0], bestLearner[1], trainingData[i])
            y_i = trainingData[i][-1]
            D[i] = D[i]*np.exp(-alpha*y_i*h_x)
        z = 1/sum(D)
        D = [x * z for x in D]
        # add bestLearner and corresponding alpha to output array
        chosenLearners.append((bestLearner, alpha))
    return chosenLearners

# return most accurate learner and its error
def selectBestLearner(trainingData, D):
    # bestLearner is of the form (i, +/-) where i is index of word to split on
    bestLearner = (-1, 0)
    epsilon = 1
    for i in range(4003):
        # errors of h_i,+ and h_i,-
        errorPlus = 0
        errorMinus = 0
        for j in range(len(trainingData)):
            line = trainingData[j]
            # output of h_i,+ and h_i,-
            hPlus = h(i, 1, line)
            hMinus = h(i, -1, line)
            # check if correct
            if hPlus != line[-1]:
                errorPlus = errorPlus + D[j]
            if hMinus != line[-1]:
                errorMinus = errorMinus + D[j]
        # update epsilon and bestLearner
        if errorPlus < epsilon:
            epsilon = errorPlus
            bestLearner = (i, 1)
        elif errorPlus == epsilon:
            rand = random.choice([i, bestLearner[0]])
            bestLearner = (rand, 1)
        if errorMinus < epsilon:
            epsilon = errorMinus
            bestLearner = (i, -1)
        elif errorMinus == epsilon:
            rand = random.choice([i, bestLearner[0]])
            bestLearner = (rand, -1)
    return (bestLearner, epsilon)

# classifier function
def h(i, sign, x):
    if sign == 1:
        if x[i] == 1:
            return 1
        else:
            return -1
    else:
        if x[i] == 1:
            return -1
        else:
            return 1

# based on input array of weak learners and data point, outputs strong learner result
def strongLearner(f, x):
    currSum = 0
    for weakL in f:
        i = weakL[0][0]
        sign = weakL[0][1]
        h_x = h(i, sign, x)
        alpha = weakL[1]
        currSum = currSum + h_x*alpha
    if currSum < 0:
        return -1
    elif currSum > 0:
        return 1
    else:
        return random.choice([1, -1])
