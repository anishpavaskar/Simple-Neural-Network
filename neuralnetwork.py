
#Simple neural network by Anish Pavaskar 2016


#imports numpy


import numpy as np  


#function definition of the sigmoid function,
def nonlin(x, deriv=False):  
    if(deriv==True):
        return (x*(1-x))
    
    return 1/(1+np.exp(-x)) 

# creates input matrix, the third column is for accommodating the bias term and is not part of the input. 


X = np.array([[0,0,1],  # Note: there is a typo on this line in the video
            [0,1,1],
            [1,0,1],
            [1,1,1]])


# The output of the exclusive OR function follows. 

# In[4]:

#output data
y = np.array([[0],
             [1],
             [1],
             [0]])


#seeds numbers to make them deterministic


np.random.seed(1)



#synapses
syn0 = 2*np.random.random((3,4)) - 1  # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
syn1 = 2*np.random.random((4,1)) - 1  # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.


#This next section is the main training loop
for j in range(60000):  
    
   
#performs matrix multiplication to predict output value
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    
 
    l2_error = y - l2
    if(j % 10000) == 0:  #only prints ever 10000 iterations 
        
        print("Error:",str(np.mean(np.abs(l2_error))))
    
    l2_delta = l2_error*nonlin(l2, deriv=True)
    
    l1_error = l2_delta.dot(syn1.T)
    
    l1_delta = l1_error * nonlin(l1,deriv=True)
    
#gradient descent alg
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    
print ("Output after training")
print (l2)
    
    




# See how the final output closely approximates the true output [0, 1, 1, 0],and the error decreases with each iteration
#If you increase the number of iterations in the training loop , the final output will be even closer. 


