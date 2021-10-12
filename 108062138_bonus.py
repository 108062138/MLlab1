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
output_dataroot = StudentID + '_bonus_prediction.csv' # Output file will be named as '[StudentID]_basic_prediction.csv'

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
    for i in range(0,np.shape(processedData)[0]):
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
    for index in  range(0, np.shape(inputT)[1]-35):
        tmp = inputT[2][index:index+7]
        reShapeTmp = tmp.reshape((1,7))
        firs = np.vstack((firs,tmp))
        ans = np.vstack((ans,inputT[2][index+8]))
        ones = np.vstack((ones,1))
    res = np.hstack((ans,firs))
    res = np.hstack((res,ones))# add this to make thte extra offset
    return res[1:,:]

def genTargetFeature():
    inputT = np.transpose(input_datalist)
    firs = np.array([1,2,3,4,5,6,7])
    ans = np.array([-1])
    ones = np.array([1])
    for index in  range(np.shape(inputT)[1]-35,np.shape(inputT)[1]-7):
        tmp = inputT[2][index:index+7]
        reShapeTmp = tmp.reshape((1,7))
        firs = np.vstack((firs,tmp))
        ans = np.vstack((ans,inputT[2][index+8]))
        ones = np.vstack((ones,1))
    res = np.hstack((ans,firs))
    res = np.hstack((res,ones))# add this to make thte extra offset
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
        arr.append(tmp*2/np.shape(processedData)[0])
    gradiant = np.array(arr)
    #print(gradiant)
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
def MakePrediction(processedData,weight):
# TODO 6: Make prediction of testing data 
    rec = 0
    for i in range(0,np.shape(processedData)[0]):
        item = processedData[i][1:9].astype(float)
        weight = weight.astype(float)
        val = np.dot(item,weight,out= None)
        print('true value',processedData[i][0],'predict value',val)

        rec+=abs(float(processedData[i][0])-float(val))/float(processedData[i][0])
    print(rec/np.shape(processedData)[0])

def genRes(processedData,weight):
    item = np.ones((8,1))
    for i in range(0,7):
        item[i]= int(processedData[i])
    #print('the item will be: ',item)
    for i in range(0,28):
        #print(item)
        weight = weight.astype(float)
        val = 0
        for j in range(0,8):
            val += item[j]* weight[j]
        #print('predict value',val)
        for j in range(0,6):
            item[j] = item[j+1]
        item[6] = val
        output_datalist.append([i,int(val)])
    with open(output_dataroot, 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Date', 'TSMC Price'])
    
        for row in output_datalist:
            writer.writerow(row)


# TODO 7: Call functions of TODO 2 to TODO 6, train the model and make prediction
if __name__ == '__main__':
    with open(input_dataroot, newline='') as csvfile:
        print('read in data')
        input_datalist = np.array(list(csv.reader(csvfile)))

    meanloss = 100000000000
    epoch = 500
    weight = np.ones((8,1),dtype=np.float64)
    weight[7] = 180 #self defined initial bias
    learningRate = 0.00000008
    processedData = PreprocessData()
    trainData, testData = SplitData(processedData)
    
    historyMin = 10000000000000
    remWeight = np.ones((8,1),dtype=np.float64)
    while epoch>0:
        weight = Regression(trainData,weight,learningRate)       
        meanloss = CountLoss(testData,weight)
        if historyMin > meanloss:
            remWeight = weight
            historyMin = meanloss
        epoch = epoch -1
        print('loss: ',meanloss)

    print('weight: ',weight)
    print('remWeight: ', remWeight)
    print('start predict')
    MakePrediction(testData,remWeight)
    print('end predict')
    print(trainData[-1])
    print(trainData[-1][1:])
    genRes(trainData[-1][1:],remWeight)
    print(output_datalist)
    

