#!/usr/bin/env python
# coding: utf-8

# import packages
# Note: You cannot import any other packages!
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import random

# Global attributes
# Do not change anything here except TODO 1 
StudentID = '108062138' # TODO 1 : Fill your student ID here
input_dataroot = 'input.csv' # Please name your input csv file as 'input.csv'
output_dataroot = StudentID + '_basic_prediction.csv' # Output file will be named as '[StudentID]_basic_prediction.csv'

input_datalist =  [] # Initial datalist, saved as numpy array
output_datalist =  [] # Your prediction, should be 20 * 2 matrix and saved as numpy array
                      # The format of each row should be [Date, TSMC_Price_Prediction] 
                      # e.g. ['2021/10/15', 512]

# You can add your own global attributes here

#hyperparameter
# From TODO 2 to TODO 6, you can declare your own input parameters, local attributes and return parameters
    
def SplitData(processedData):
# TODO 2: Split data, 2021/10/15 ~ 2021/11/11 for testing data, and the other for training data and validation data 
# handle 28days
    trainLIst = []
    testList = []
    for i in range(0,np.shape(processedData)[1]):
        if i%4 == 0:
            testList.append(processedData[i])
        else:
            trainLIst.append(processedData[i])
    trainData = np.array(trainLIst)
    testData = np.array(testList)
    return [trainData,testData]
def PreprocessData():
# TODO 3: Preprocess your data  e.g. split datalist to x_datalist and y_datalist
    inputT = np.transpose(input_datalist)
    firs = np.array([1,2,3,4,5,6,7])
    ans = np.array([-1])
    ones = np.array([1])
    for index in  range(1, np.shape(inputT)[1]-35):
        tmp = inputT[1][index:index+7]
        reShapeTmp = tmp.reshape((1,7))
        firs = np.vstack((firs,tmp))
        ans = np.vstack((ans,inputT[2][index]))
        ones = np.vstack((ones,1))
    res = np.hstack((ans,firs))
    res = np.hstack((res,ones))# add this to make thte extra offset
    print(np.shape(res))
    return res[1:,:]


def Regression(processedData,weight,learningRate):
# TODO 4: Implement regression
    arr = []
    for i in range(0,np.shape(processedData)[0]):
        item = processedData[i][1:9].astype(float)
        weight = weight.astype(float)
        val = np.dot(item,weight,out= None)
        arr.append(val)
    capY = np.array(arr)
    arr = []
    for j in range(0,8,1):
        tmp = 0 
        for i in range(0,np.shape(processedData)[0]):
            tmp -= (processedData[i][0].astype(float)-capY[i].astype(float)) * processedData[i][j].astype(float)# the phi function is just x
        arr.append(tmp*2)
    gradiant = np.array(arr)
    print(gradiant)
    newWeight = np.subtract(weight,gradiant * learningRate)
    return newWeight

def CountLoss(processedData,weight):#return mse square
# TODO 5: Count loss of training and validation data
    arr = []
    for i in range(0,np.shape(processedData)[0]):
        item = processedData[i][1:9].astype(float)
        weight = weight.astype(float)
        val = np.dot(item,weight,out= None)
        arr.append(val)
    capY = np.array(arr)
    cnt = 0
    for i in range(0,np.shape(processedData)[0]):
        tmp = capY[i].astype(float)-processedData[i][0].astype(float)
        cnt += tmp**2# produce mse
    return cnt/np.shape(processedData)[0]
def MakePrediction():
# TODO 6: Make prediction of testing data 
    print('hehe')


# TODO 7: Call functions of TODO 2 to TODO 6, train the model and make prediction
if __name__ == '__main__':
    # Write prediction to output csv
    # Read input csv to datalist
    with open(input_dataroot, newline='') as csvfile:
        print('read in data')
        input_datalist = np.array(list(csv.reader(csvfile)))
    
    learningRate = 0.000001
    processedData = PreprocessData()
    testData, trainData = SplitData(processedData)

    meanloss = 100000000000
    epoch = 500
    weight = np.zeros((8,1),dtype=np.float64)
    while meanloss>100000 and epoch>0:
        weight = Regression(processedData,weight,learningRate)
        meanloss = CountLoss(trainData,weight)
        epoch = epoch -1
        print(meanloss)
    print(weight)
    
    #CountLoss()
    #MakePrediction()
    #print('end predict')
    #with open(output_dataroot, 'w', newline='', encoding="utf-8") as csvfile:
    #    writer = csv.writer(csvfile)
    #    writer.writerow(['Date', 'TSMC Price'])
    #
    #    for row in output_datalist:
    #        writer.writerow(row)

