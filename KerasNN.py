#Team Members:
# Dhruv Bajpai - dbajpai - 6258833142
# Anupam Mishra - anupammi - 2053229568

import sys
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.initializers import RandomUniform
from keras import optimizers
import numpy as np
from sklearn.metrics import accuracy_score as Accuracy
from PIL import Image
np.random.seed(0)
np.set_printoptions(threshold=np.nan)

def getImageData(dataPaths):
    data = np.array([np.array(Image.open(x),dtype=np.float) for x in dataPaths])
    return data

def getAllTrainingData(filename):
    trainingDataPaths = []
    Y = []
    with open(filename,"r") as trainFile:
        for path in trainFile:
            trainingDataPaths.append(path[:-1])
            if "down" in path:
                Y.append(1)
            else:
                Y.append(0)
            
    data = getImageData(trainingDataPaths)
    #flattening data
    data = data.reshape(-1,30*32)
    X = data/255
    Y = np.array(Y)
    return X,Y

def main():
    X_train, Y_train = getAllTrainingData(sys.argv[1])
    X_test, Y_test = getAllTrainingData(sys.argv[2])
    model = Sequential()
    model.add(Dense(100, kernel_initializer=RandomUniform(minval = -0.01, maxval = 0.01, seed = None), input_shape=(960,), activation='sigmoid'))
    model.add(Dense(1,kernel_initializer=RandomUniform(minval = -0.01, maxval = 0.01, seed = None), activation='sigmoid'))
    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    model.fit(X_train,Y_train,epochs=1000)
    outputData = model.predict(X_test)
    outputData[outputData<0.5] = 0
    outputData[outputData>0.5] = 1
    accuracy = Accuracy(outputData,Y_test)
    print("---------------- Predictions -----------------\n")
    output_path = []
    with open(sys.argv[2],"r") as trainFile:
        for path in trainFile:
            output_path.append(path[:-1])
    for i in range(0,len(output_path)):
        print (output_path[i],"==>  ",np.asscalar(outputData[i]))
    
    print("---------------- Accuracy --------------------")
    print(accuracy*100,"%")

if __name__ == '__main__':
    main()