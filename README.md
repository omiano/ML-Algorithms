# ML-Algorithms

Implementations of the k-Nearest Neighbor, ID3 Decision Tree with and without pruning, linear and kernelized Perceptron, and Boosting classifier algorithms.

## Motivation

I wanted to implement popular machine learning algorithms to compare their run times and error rates.

## How To Use

### k-Nearest Neighbor
Call the **kNN** function with the parameter *vector* being the feature vector you want to predict a label for, *data* being a matrix of feature vectors to train the classifier on, *labels* a vector of the training data's labels, and *k* the number of neighbors you would like to use for the algorithm. The function will return the label it predicts.

### ID3 Decision Tree

#### Without Pruning

Call the **id3** function with the parameter *trainingData* being a matrix of feature vectors to train the classifier on. The function will return the root node of a decision tree where each leaf node's data is the label to predict and its yes and no pointers are null, and every other node's data is of the form *[value, index]*. Starting at the root node, if the value at *index* of the feature vector you want to predict a label for is less than or equal to *value*, follow the yes pointer. Otherwise, follow the no pointer. Repeat until you get to a leaf node.

#### With Pruning

Call the **id3** function with the parameter *trainingData* being a matrix of feature vectors to train the classifier on. Then call the **pruneTree** function with the parameter *rootNode* being the root node returned by **id3**. The function will prune the decision tree one time. Starting at the root node, if the value at *index* of the feature vector you want to predict a label for is less than or equal to *value*, follow the yes pointer. Otherwise, follow the no pointer. Repeat until you get to a leaf node.

### Perceptron

#### Linear

Call the **perceptron** function with the parameter *trainingData* being a matrix of feature vectors to train the classifier on, and *numPasses* the number of passes you would like it to perform on the data. The function will return a vector of weights, *w*. If the dot product of *w* and feature vector you want to predict a label for is less than 0, predict 2. If it is greater than 0, predict 1. If it is equal to 0, predict either 2 or 1 randomly.

#### Kernelized

Call the **kernelizedPerceptron** function with the parameter *trainingData* being a matrix of feature vectors to train the classifier on, *numPasses* the number of passes you would like it to perform on the data, and *K* the kernel function you would like to use. The function will return a matrix *M*. For each entry *i* in *M*, calculate the sum of *y[i]K(x[i], x)* where *x* is the *i*th feature vector and *y* is its corresponding label. If the sum is less than 0, predict 2. If it is greater than 0, predict 1. If it is equal to 0, predict either 2 or 1 randomly.

### Boosting

Call the **boost** function with the parameter *trainingData* being a matrix of feature vector sto train the classifier on and *numRounds* being the number of rounds of boosting you would like to perform. The function will return an array of the weak learners chosen for each round. Then call the **strongLearner** function with the parameter *f* being the array of weak learners and *x* the feature vector you want to predict a label for. The function will return the predicted label.

## Details

The candidate weak learners I used for the Boosting algorithm were *h1_i(x) = 1 if index i of the feature vector is 1, -1 otherwise* and *h2_i(x) = 1 if index i of the feature vector is 0, -1 otherwise*.
