#Team Members:
# Dhruv Bajpai - dbajpai - 6258833142
# Anupam Mishra - anupammi - 2053229568

import numpy as np
from PIL import Image
np.random.seed(1)
np.set_printoptions(threshold=np.nan)
from sklearn.metrics import accuracy_score as Accuracy

def getImageData(dataPaths):
    data = np.array([np.array(Image.open(x),dtype=np.float) for x in dataPaths])
    return data

def getAllData(fileName):
    trainingDataPaths = []
    Y = []
    with open(fileName,"r") as trainFile:
        for path in trainFile:
            trainingDataPaths.append(path[:-1])
            if "down" in path:
                Y.append(1)
            else:
                Y.append(0)
            
    data = getImageData(trainingDataPaths)
    #flattening data
    data = data.reshape(-1,30*32)
    data/=255
    #appending intial column to data
    vector = np.ones([data.shape[0],1])
    X = np.concatenate((vector,data),axis=1)
    Y = np.array(Y)
    return X,Y

X_train, Y_train = getAllData("downgesture_train.list")
X_test,Y_test = getAllData("downgesture_test.list")

def forwardProp(X,theta1,theta2):
#     Take in the single input image and returns the predicted value after passing through the network
    l1 = matMultiply(theta1,X)
    a1 = sigmoid(l1)
    vector = np.ones([1,1])
    a1New = np.concatenate((vector,a1),axis=0)
    l2 = matMultiply(theta2,a1New)
    return a1New, sigmoid(l2)

def backProp(a1,l2,X,Y):
    deltaL = 2 *(l2-Y) * l2 * (1-l2)
    theta2Updates = matMultiply(deltaL,a1.transpose())
    deltaHiddenLayer = matMultiply(theta2.transpose(),deltaL) * (a1* (1-a1))
    deltaHiddenLayer = deltaHiddenLayer[1:,:]
    theta1Updates = matMultiply(deltaHiddenLayer,X.transpose())
    return theta1Updates, theta2Updates
    
def matMultiply(a,b):
    return np.dot(a,b)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

epochCount = 1000
learningRate = 0.1
theta1 = np.random.uniform(low=-0.01,high=0.01,size=(100,961))
theta2 = np.random.uniform(low=-0.01,high=0.01,size=(1,101))

print ("Beginning Training....")
# Training Implementation
for j in range(0,epochCount):
    if j%100==0:
        print ("Epoch Number: {} ".format(j))
    for i in range(0,X_train.shape[0]):
        X = X_train[i].reshape(961,-1)
        Y = Y_train[i]
        a1, l2 = forwardProp(X,theta1,theta2)
        theta1Updates, theta2Updates = backProp(a1,l2,X,Y)
        theta1 = theta1 -(learningRate * theta1Updates)
        theta2 = theta2 - (learningRate *  theta2Updates)
print ("Done Training!!")
    
Y_test_prediction = np.random.rand(X_test.shape[0],1)

for i in range(0,X_test.shape[0]):
    a1new, curPrediction = forwardProp(X_test[i].reshape(961,-1),theta1,theta2)
    Y_test_prediction[i][0] = curPrediction

Y_test_prediction[Y_test_prediction<0.5]=0
Y_test_prediction[Y_test_prediction>0.5]=1

output_path = []
with open("downgesture_test.list","r") as trainFile:
        for path in trainFile:
            output_path.append(path[:-1])
for i in range(0,len(output_path)):
    print (output_path[i],"==>  ",np.asscalar(Y_test_prediction[i]))


count = 0
for i in range(0,Y_test.shape[0]):
    if Y_test[i]!=Y_test_prediction[i][0]:
        count+=1
print ("Number of Misclassifications: {}".format(count))

accuracy = Accuracy(Y_test_prediction,Y_test)
print ("Accuracy: ",accuracy*100," %")