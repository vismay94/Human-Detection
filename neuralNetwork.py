import numpy as np

def createNeuralNetwork(X,Y,no_hidden_neurons):
    # X = input_data of 20 images
    # Y= Result of image, Positive = 1 and Negative = 0. Pos
    # no_hidden_neurons represents total nodes at hidden layer
    #np.random.seed(1)
    np.random.seed(1)
    w1 = np.random.randn(no_hidden_neurons, len(X[0])) * 0.01
    b1 = np.zeros((no_hidden_neurons,1))
    w2 = np.random.randn(1,no_hidden_neurons) * 0.01
    b2 = np.zeros((1,1))
    
    
    dictionary = {}
    cost_avg = 0.0 
    old_cost = 0.0

    for i in range(0,200):
        cost = 0.0
        for j in range(0,len(X)):
            features = X[j].shape[0]
            q = X[j].reshape(1,features)
            q = q.T
            '''Neural network train'''
            v1 = w1.dot(q)+ b1   #Multiplication for Level 1 hidden layer
            a1 = ReLu(v1)
            v2 = w2.dot(a1) + b2
            a2 = sigmoid(v2)
            cost += (1.0/2.0)*(np.square((a2-Y[j])))
            #print(a2)
            #print("Cost [",j,"] = ",cost)
            
            # Backward Propogation
            diff2 = (a2-Y[j])  *  derSigmoid(a2)    #find the differene in value
            dw2 = np.dot(diff2,a1.T)
            db2 = np.sum(diff2,axis=1, keepdims=True)

            diff1 = w2.T.dot(diff2) * ReLuDerivation(a1)            
            dw1 =  np.dot(diff1,q.T)
            db1 =  np.sum(diff1,axis=1, keepdims=True)

            #updating weights
            w1 = w1 - 0.01*dw1
            w2 = w2 - 0.01*dw2
            b1 = b1 - 0.01*db1
            b2 = b2 - 0.01*db2
            # dictionary = {'w1':w1,'b1':b1,'w2':w2,'b2':b2}
            #dictionary = {'w1':w1,'w2':w2}
            # '''End of neural network'''
            # cost += (abs(a2-Y[j]))**2
        cost_avg = cost/len(X)
        print("Epoch = ",i,"cost_avg = ",cost_avg[0][0])
        dictionary = {'w1':w1,'b1':b1,'w2':w2,'b2':b2} #save our updated weights. So that we can use them while testing.
        # if cost between two epochs is less than 0.01, we will stop. Because we know that our weights does not change too much.
        if(abs(old_cost-cost_avg)<=0.0001):  # 0.0001   
            return dictionary
        else:
            old_cost = cost_avg

    return dictionary

def saveDictionary(dictionary,name):
    # Saving Dictionary Calculated Values for Future Reference If need arises
    np.save(str(name)+".npy",dictionary)
    print("Successfully saved model file as",str(name),".npy")

def loadDictionary(name):
    #This function will load already calculated dictionary value          
    dictionary = np.load(str(name)+".npy")
    return dictionary[()]

def predict(X_test,dictionary):
    # Predict the newly uploaded images.
    # X_test= new image to identify.
    # dictionary= containing our trained model parameters
    features = X_test.shape[0]
    q = X_test.reshape(1,features)
    q = q.T

    w1,w2,b1,b2 = dictionary['w1'],dictionary['w2'],dictionary['b1'],dictionary['b2'] # getting the data from the dictionary.
    z1 = w1.dot(q)+ b1   
    a1 = ReLu(z1)
    z2 = w2.dot(a1) + b2
    a2 = sigmoid(z2)
    print(a2)
    return a2


# Return value between 0 and 1.
def sigmoid(val):
    return 1.0 / (1.0 + np.exp(-val))

def derSigmoid(x):
    return x * (1-x)

# For ForwardFeed, It return original value if it's >0 , or 0 if value
def ReLu(val):
    return val*(val>0)

# In Backward Propogration, It return either 1 or 0
def ReLuDerivation(x):
    return 1. * (x > 0)

# dictionary = createNeuralNetwork(X,Y,250)
