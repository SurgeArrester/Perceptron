# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 22:28:39 2018

@author: Cameron Hargreaves
"""

import numpy as np
import csv

def main():
    firstClass = 1
    secondClass = 2
    thirdClass = 3
    
    numEpochs = 20 
    
    inputFileTrain = readDataFile('train.data')
    inputFileTest = readDataFile('test.data')

    trainSet = [filterInput(np.copy(inputFileTrain), firstClass, secondClass),   # Create list for each set to train each set of weights
                filterInput(np.copy(inputFileTrain), firstClass, thirdClass), 
                filterInput(np.copy(inputFileTrain), secondClass, thirdClass)]
            
    weights = [trainWeights(trainSet[0], numEpochs),            # train weights for each class
               trainWeights(trainSet[1], numEpochs),
               trainWeights(trainSet[2], numEpochs)]
    
    trainPrediction = [perceptronPredict(inputFileTrain, weights[0]),   # Create +/- 1 prediction arrays for each set of two classes
                       perceptronPredict(inputFileTrain, weights[1]),
                       perceptronPredict(inputFileTrain, weights[2])]
    
    testPrediction = [perceptronPredict(inputFileTest, weights[0]),
                      perceptronPredict(inputFileTest, weights[1]),
                      perceptronPredict(inputFileTest, weights[2])]
    
    trainPredFull = multiclassPred(np.copy(trainPrediction))    # merge the three arrays by taking the mode 
    testPredFull = multiclassPred(np.copy(testPrediction))
    
    accuraciesTrain = multiclassAccuracy(inputFileTrain, trainPredFull) # Compare against the original array
    accuraciesTest = multiclassAccuracy(inputFileTest, testPredFull)
    
    print("For the training data:\nClass 1: {}% accuracy\nClass 2: {}% accuracy\nClass 3: {}% accuracy\n".format((accuraciesTrain[0]/(accuraciesTrain[0] + accuraciesTrain[1])) * 100, (accuraciesTrain[2]/(accuraciesTrain[2] + accuraciesTrain[3])) * 100, (accuraciesTrain[4]/(accuraciesTrain[4] + accuraciesTrain[5])) * 100))
    print("For the Testing data:\nClass 1: {}% accuracy\nClass 2: {}% accuracy\nClass 3: {}% accuracy".format((accuraciesTest[0]/(accuraciesTest[0] + accuraciesTest[1])) * 100, (accuraciesTest[2]/(accuraciesTest[2] + accuraciesTest[3])) * 100, (accuraciesTest[4]/(accuraciesTest[4] + accuraciesTest[5])) * 100))

    return accuraciesTrain, accuraciesTest

def readDataFile(fileName):
    ''' Take a filename and return the csv as a 2d list '''
    inputFile = np.genfromtxt(fileName, delimiter=',') # Read in csv
    rowCount = inputFile.shape[0]
    with open(fileName, 'r') as csvfile:    # Read in the csv again and replace the final column to just have numbers
        csvReader = csv.reader(csvfile, delimiter=',')
        for row, i in zip(csvReader, range(rowCount)):
            inputFile[i, 4] = row[4][6:]
    return inputFile

def filterInput(inputFile, firstClass = 1.0, secondClass = 2.0):
    '''Replace the original classes with ones and minus ones'''
    rowToDelete = []
    for row, i in zip(inputFile, range(inputFile.shape[0])):
        if row[4] == firstClass:
            inputFile[i][4] = -1
        elif row[4] == secondClass:
            inputFile[i][4] = 1
        else:
            rowToDelete.append(i)
    filteredMatrix = np.delete(inputFile, rowToDelete, 0)
    return filteredMatrix

def trainWeights(train, numEpochs):
    '''
    Update the weights using the perceptron algorithm
    
    This initialises a weight vector, shuffles the training set, makes a prediction
    on the existing weights and row, computes the difference from the actual (error)
    then updates each of the weights via the perceptron algorithm
    '''
    weights = [0.0 for i in range(len(train[0]))]  # initialise weights to be 0
    for epoch in range(numEpochs):
        np.random.shuffle(train)
#        print(weights)
        for row in train:
            sumError = 0.0
            prediction = perceptron(row[:-1], weights)
            error = (row[-1] - prediction) / 2  # will be zero on correct prediction, +/- 1 otherwise
            sumError += error**2

            weights[0] = weights[0] + error # Update the bias
            for i in range(1, len(row)):    # Update the weights for each of the inputs
                weights[i] = weights[i] + error * row[i - 1]
    return weights

def perceptron(row, weights):
    '''
    Activate threshold based on the weight and input vectors
    
    We assume the weight vector is of length one greater than the input vector and
    bias is in the first column of the weights vector
    '''
    activation = 0
    for i in range(len(row)):
        activation += weights[i + 1] * row[i]       
    return 1 if activation + weights[0] >= 0 else -1

def perceptronPredict(inputArray, weights):
    '''
    Return an array of predictions based on inputs and weights    
    '''
    predictionArray = np.empty(inputArray.shape[0])
    for i in range(inputArray.shape[0]):
        predictionArray[i] = perceptron(inputArray[i][:-1], weights)
    return predictionArray

def generateTrainingData(numRows, meanA, meanB):
    '''
    Return two arrays of equal numbers of classified gaussian distributions with variation 1
    '''
    fullSet = np.zeros((numRows, 2))

    for i in range(int(numRows/2)):
        fullSet[i] = np.array([np.random.normal(meanA, 1), -1])    # set gaussians for our two classes
        fullSet[i + int(numRows/2)] = np.array([np.random.normal(meanB, 1), 1])
    
    trainSet = np.concatenate((fullSet[:int(numRows * 0.4)], fullSet[int(numRows * 0.5):int(numRows * 0.9)]))   # take 80% of the values and put into the training
    testSet = np.concatenate((fullSet[int(numRows * 0.4):int(numRows * 0.5), :], fullSet[int(numRows * 0.9):numRows, :])) # take rest and make testing
    return trainSet, testSet
    
def confusionMatrix(inputArray, prediction):
    '''
    Return a list of [truePositives, falsePositives, trueNegatives, falseNegatives]
    '''
    truePositives = 0
    falsePositives = 0
    trueNegatives = 0
    falseNegatives = 0
    
    for i, row in enumerate(inputArray):
        if row[-1] == 1 and prediction[i] == 1:
            truePositives += 1
        elif row[-1] == -1 and prediction[i] == 1:
            falsePositives += 1
        elif row[-1] == 1 and prediction[i] == -1:
            falseNegatives += 1
        elif row[-1] == -1 and prediction[i] == -1:
            trueNegatives += 1
    return [truePositives, falsePositives, trueNegatives, falseNegatives]

'''
Below are basic functions for standard measures of classifier evaluation
'''

def accuracy(confusionMatrix):
    return (confusionMatrix[0] + confusionMatrix[2]) / sum(confusionMatrix)

def precision(confusionMatrix):
    return confusionMatrix[0] / (confusionMatrix[0] + confusionMatrix[1])

def recall(confusionMatrix):
    return confusionMatrix[0] / (confusionMatrix[0] + confusionMatrix[3])

def falsePositiveRate(confusionMatrix):
    return confusionMatrix[1] / (confusionMatrix[1] + confusionMatrix[2])

def fScore(confusionMatrix):
    return (2 * precision(confusionMatrix) * recall(confusionMatrix)) / (precision(confusionMatrix) + recall(confusionMatrix))

def scores(inputArray, prediction):
    confMat = confusionMatrix(inputArray, prediction)
    return [accuracy(confMat), precision(confMat), recall(confMat), falsePositiveRate(confMat), fScore(confMat)]

def multiclassPred(prediction):
    for row in range(prediction[0].shape[0]):   # loop through each row and replace for associated classes
        if prediction[0][row] < 0:              # TODO replace with lambdas
            prediction[0][row] = 1
        else:
            prediction[0][row] = 2
            
        if prediction[1][row] < 0:
            prediction[1][row] = 1
        else:
            prediction[1][row] = 3
            
        
        if prediction[2][row] < 0:
            prediction[2][row] = 2
        else:
            prediction[2][row] = 3
            
    mergedPrediction = []
    for row in range(prediction[0].shape[0]):
        predictions = [prediction[0][row], prediction[1][row], prediction[2][row]]
        mergedPrediction.append(max(set(predictions), key = predictions.count))
    return mergedPrediction

def multiclassAccuracy(inputData, predictedClass):
    sumCorrectOne = 0
    sumFalseOne = 0
    sumCorrectTwo = 0
    sumFalseTwo = 0
    sumCorrectThree = 0
    sumFalseThree = 0 
    
    for i, row in enumerate(inputData):
        if row[-1] == 1 and row[-1] == predictedClass[i]:
            sumCorrectOne += 1
        elif row[-1] == 1 and row[-1] != predictedClass[i]:
            sumFalseOne += 1
            
        if row[-1] == 2 and row[-1] == predictedClass[i]:
            sumCorrectTwo += 1
        elif row[-1] == 2 and row[-1] != predictedClass[i]:
            sumFalseTwo += 1
            
        if row[-1] == 3 and row[-1] == predictedClass[i]:
            sumCorrectThree += 1
        elif row[-1] == 3 and row[-1] != predictedClass[i]:
            sumFalseThree += 1
            
    return [sumCorrectOne, sumFalseOne, sumCorrectTwo, sumFalseTwo, sumCorrectThree, sumFalseThree]

train, test = main()

for i in range(100):
    train2, test2 = main()
    for i in range(len(train)):
        train[i] += train2[i]
        test[i] += test2[i]
    
averagetrain = [x / 100 for x in train]
averagetest = [x / 100 for x in test]

print("For the training data:\nClass 1: {}% accuracy\nClass 2: {}% accuracy\nClass 3: {}% accuracy\n".format((averagetrain[0]/(averagetrain[0] + averagetrain[1])) * 100, (averagetrain[2]/(averagetrain[2] + averagetrain[3])) * 100, (averagetrain[4]/(averagetrain[4] + averagetrain[5])) * 100))
print("For the testing data:\nClass 1: {}% accuracy\nClass 2: {}% accuracy\nClass 3: {}% accuracy\n".format((averagetest[0]/(averagetest[0] + averagetest[1])) * 100, (averagetest[2]/(averagetest[2] + averagetest[3])) * 100, (averagetest[4]/(averagetest[4] + averagetest[5])) * 100))


