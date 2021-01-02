## ---------------------------- ##
##
## Example student submission code for autonomous driving challenge.
## You must modify the train and predict methods and the NeuralNetwork class. 
## 
## ---------------------------- ##

import numpy as np
import cv2
from tqdm import tqdm
import time

def alv_vision(image, rgb, thresh):
    return (np.dot(image.reshape(-1, 3), rgb) > thresh).reshape(image.shape[0], image.shape[1])

def train(path_to_images, csv_file):
    '''
    First method you need to complete. 
    Args: 
    path_to_images = path to jpg image files
    csv_file = path and filename to csv file containing frame numbers and steering angles. 
    Returns: 
    NN = Trained Neural Network object 
    '''

    # You may make changes here if you wish. 
    # Import Steering Angles CSV
    data = np.genfromtxt(csv_file, delimiter = ',')
    frame_nums = data[:,0]
    steering_angles = data[:,1]
    train = []
    bins_values = []
    NN = NeuralNetwork()
    bins=NN.bins
    
    # You could import your images one at a time or all at once first, 
    # here's some code to import a single image:
    for i in range(1500):
        image = cv2.imread(path_to_images + '/' + str(int(i)).zfill(4) + '.jpg')
        res = cv2.resize(image, dsize=(64, 60), interpolation=cv2.INTER_AREA)
        final_X = alv_vision(res,rgb = [1, 0, -1],thresh = 0.75)
        before_X= np.array(final_X).flatten()/255.0
        train.append(before_X)
        steering_angles = np.array(steering_angles)
        bins_values_y = np.digitize(steering_angles[i],bins)
        bins_values_y -= 1
        trail = np.zeros(bins.shape[0])
        trail = np.zeros(60)
        trail = trail.tolist()
        list1 = [1.0,0.9,0.6,0.4,0.2]
        try:
            for i in range(5):
                check = bins_values_y-i
                trail[bins_values_y+i] = list1[i]
                if check<0:
                    pass
                else:
                    trail[bins_values_y-i] = list1[i]
    
        except IndexError:
            trash=0
        print(trail)
        print("the values of trail are",trail)
        bins_values.append(trail)
    data = np.asanyarray(train)
    y_data = np.asanyarray(bins_values)
    for i in range(300):
        total_count = data.shape[0]
        max_batch = 50
        chunks = (total_count-1)//max_batch+1
        for j in range(chunks):
            X = data[j*50:(j+1)*50,:]
            print(X.shape)
            y = y_data[j*50:(j+1)*50]
            print(y.shape)
            params = NN.getParams()
            #dJdW1,dJdW2 = NN.costFunctionPrime(X,y)
            grads = NN.computeGradients(X,y)
            #NN.W1 = NN.W1 - 0.0001 * dJdW1
            #NN.W2 = NN.W2 - 0.0001 * dJdW2
            set_params = params - 0.4*grads
            NN.setParams(set_params)
    return NN


def predict(NN, image_file):
    '''
    Second method you need to complete. 
    Given an image filename, load image, make and return predicted steering angle in degrees. 
    '''
    im_full = cv2.imread(image_file)
    res = cv2.resize(im_full, dsize=(64, 60), interpolation=cv2.INTER_AREA)
    final_X = alv_vision(res,rgb = [1, 0, -1],thresh = 0.75)
    before_X= np.array(final_X).flatten()/255.0
    bins = NN.bins
    final_bins = np.array(bins)
    #print(final_bins)
    #print(final_bins.shape)
    X= before_X.reshape(1,before_X.shape[0])
    angle = NN.forward(X)
    angle= angle.reshape(60)
    angle_index = np.argmax(angle)
    predicted_angle = final_bins[angle_index]
    ## Perform inference using your Neural Network (NN) here.
    #print(predicted_angle)
    return predicted_angle

class NeuralNetwork(object):
    def __init__(self):        
        '''
        Neural Network Class, you may need to make some modifications here!
        '''
        self.inputLayerSize = 3840
        self.outputLayerSize = 60
        self.hiddenLayerSize = 10

        self.bins = np.linspace(-170,25,60)
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
    
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*np.sum((y-self.yHat)**2)
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2
    
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
