import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle
import os

'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark

Since the available data is less and there is class imbalance the data is partitioned in such a way so that the model will be able to learn the representations of the data and learn to differentiate
between both the classes.

Loss function : Binary crossentropy which outputs a scalar which is the average cost over all the training samples

Weights and Biases : initialized randomly from standard normal distribution

Forward propagation :
linear combination and activations at each layer is cached so that they can be retrieved while calculating gradients and updating the parameters so that recomputation is avoided.Sigmoid is used as the activation function.

Hyperparameters:

learning rate : 0.01
epochs : 15
number of layers : 2 (one hidden layer , one output layer)
number of neurons in the first hidden layer : 7
number of neurons in the output layer : 1

Backprop:

Batch gradient descent is used for updating the weights while performing backpropagation

Results:

Confusion Matrix : 
[[2, 1], [2, 17]]
Accuracy :  0.8636363636363636
Precision : 0.9444444444444444
Recall : 0.8947368421052632
F1 SCORE : 0.918918918918919
'''

class NN:

    ''' X and Y are dataframes '''
    def __init__(self,neurons_list,epochs:int,lr:float):
        self.neurons_each_layer = neurons_list
        self.weights=list(range(len(neurons_list)))
        self.biases=list(range(len(neurons_list)))
        self.n_layers=len(neurons_list)
        self.epochs = epochs
        self.lr = lr
        
    def split_data(self,lbw):
        lbw_0 = lbw[lbw['Result'] == 0]
        lbw_1 = lbw[lbw['Result'] == 1]
        l0 = len(lbw_0)
        l1 = len(lbw_1)
        train0 = lbw_0.sample(int(0.85*l0))
        test0 = lbw_0[~lbw_0.index.isin(train0.index)]
        train1 = lbw_1.sample(int(0.75*l1))
        test1 = lbw_1[~lbw_1.index.isin(train1.index)]
        train = train0.append(train1)
        test = test0.append(test1)
        return (train,test)

    def compute_loss(self,y_hat,y_test):
        loss = - np.sum(y_test*np.log2(y_hat) + (1 - y_test)*np.log2(1 - y_hat))/len(y_hat)
        return loss
    
        
    def fit(self,X_train,y_train):    
        '''
        Function that trains the neural network by taking x_train and y_train samples as input
        '''
        #print(X_train.shape[1])
        n_samples = X_train.shape[0]
        #print("Number of samples : ",n_samples)
        self.weights[0] = np.random.randn(self.neurons_each_layer[0],X_train.shape[-1])
        self.biases[0] = np.random.randn(self.neurons_each_layer[0],1)
        for i in range(1,self.n_layers):
            self.weights[i] = np.random.randn(self.neurons_each_layer[i],self.neurons_each_layer[i-1])
            self.biases[i] = np.random.randn(self.neurons_each_layer[i],1)

        
        for _ in range(self.epochs):
            y_train = y_train.reshape(-1,1).T
            #print(y_train.shape)
            
            zs=[]
            activations=[X_train.T]
            deltas = list(range(self.n_layers))
            for i in range(self.n_layers):
                z = np.dot(self.weights[i],activations[-1]) + self.biases[i]
                #print(f"Linear Combination shape layer {i+1} : ",z.shape)
                zs.append(z)
                activation = self.sigmoid(z)
                activations.append(activation)
                #print(f"Activations shape layer {i+1} : ",z.shape)
            
            deltaL = (activations[-1] - y_train)/n_samples
            deltas[-1] = deltaL
            #print("Output Layer delta shape : ",deltaL.shape)
            dJ_dbL = np.sum(deltaL,axis=1,keepdims=True)
            #print("Nabla b output layer shape : ",dJ_dbL.shape)
            dJ_dWL = np.dot(deltaL,activations[-2].T)/n_samples
            #print("Nable W for output layer shape : ",dJ_dWL.shape)
            self.weights[-1] = self.weights[-1] - self.lr*dJ_dWL
            self.biases[-1] = self.biases[-1] - self.lr*dJ_dbL
            
            for l in range(self.n_layers-1,0,-1):
                delta = np.dot(self.weights[l].T,deltas[l]) * self.sigmoid(zs[l-1],derivative=True)
                #print('Delta shape : ',delta.shape)
                deltas[l-1]=delta
                dJ_db = np.sum(delta,axis=1,keepdims=True)
                #print(f"Nabla b shape layer {l} : ",dJ_db.shape)
                dJ_dW = np.dot(delta,activations[l-1].T)/n_samples
                #print(f"Nabla W shape layer {l} : ",dJ_dW.shape)
                self.weights[l-1] = self.weights[l-1] - self.lr*dJ_dW 
                self.biases[l-1] = self.biases[l-1] - self.lr*dJ_db 
                
                
            
            
    def predict(self,X):
        """
        The predict function performs a simple feed forward of weights
        and outputs yhat values yhat is a list of the predicted value for df X
        """
        a = X.T
        for i in range(self.n_layers):
            z = np.dot(self.weights[i],a) + self.biases[i]
            a = self.sigmoid(z)
        
        return a
    
    def sigmoid(self,inputs,derivative=False):
        if not derivative:
            return 1/(1 + np.exp(-inputs))
        return self.sigmoid(inputs)*(1 - self.sigmoid(inputs))

    def CM(self,y_test,y_test_obs):
        '''
        Prints confusion matrix 
        y_test is list of y values in the test dataset
        y_test_obs is list of y values predicted by the model
        '''
        for i in range(len(y_test_obs)):
            if(y_test_obs[i]>0.6):
                y_test_obs[i]=1
            else:
                y_test_obs[i]=0
        
        cm=[[0,0],[0,0]]
        fp=0
        fn=0
        tp=0
        tn=0
        
        for i in range(len(y_test)):
            if(y_test[i]==1 and y_test_obs[i]==1):
                tp=tp+1
            if(y_test[i]==0 and y_test_obs[i]==0):
                tn=tn+1
            if(y_test[i]==1 and y_test_obs[i]==0):
                fp=fp+1
            if(y_test[i]==0 and y_test_obs[i]==1):
                fn=fn+1
        cm[0][0]=tn
        cm[0][1]=fp
        cm[1][0]=fn
        cm[1][1]=tp
        p= tp/(tp+fp)
        r=tp/(tp+fn)
        f1=(2*p*r)/(p+r)
    
        print("Confusion Matrix : ")
        print(cm)
        print("\n")
        print("Accuracy : ", (cm[0][0] + cm[1][1])/len(y_test_obs))
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")



if __name__ == '__main__' :
    m = NN([7,1],15,0.01)
    os.chdir('..')
    preprocessed = pd.read_csv('data/pre_processed.csv')
    lbw = shuffle(preprocessed)
    train,test = m.split_data(lbw)
    X_train = train.iloc[:,:-1].values
    y_train = train.iloc[:,-1].values
    X_test = test.iloc[:,:-1].values
    y_test = test.iloc[:,-1].values
    #print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
    m.fit(X_train,y_train)
    yhat = m.predict(X_test)
    #print(yhat[0])
    m.CM(y_test,yhat[0])
    
