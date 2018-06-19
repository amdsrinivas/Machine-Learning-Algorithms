from __future__ import division
import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))
# inputs
features = np.asarray([1,0,1])
targets = np.asarray([1])
#print(features)

# hidden layer with 3 units

layer1_weights = np.random.randn(2,3)
layer1_biases = np.random.normal(size=2)
layer1_outputs = np.zeros(2)
layer1_errors = np.zeros(2)

#print(layer1_biases, layer1_weights, layer1_outputs, layer1_errors, sep='\n')

# output layer with 1 unit.
outlayer_weights = np.random.randn(1,2)
outlayer_biases = np.random.normal(size=1)
outlayer_outputs = np.zeros(1)
outlayer_errors = np.zeros(1)

#print(outlayer_biases, outlayer_weights, outlayer_outputs, outlayer_errors, sep='\n')

# weights and bias tuning.
layer1_weight_corrections =  np.zeros(shape=(2,3))
layer1_bias_corrections  = np.zeros(2)
outlayer_weight_corrections = np.zeros(shape=(1,2))
outlayer_bias_corrections = np.zeros(1)

# learning rate and number of steps.
learning_rate = 0.1
steps = 1500
k = 0
while (k < steps):
    # output calculation at hidden layer.

    for node in np.ndindex(layer1_outputs.shape):
        layer1_outputs[node] = sigmoid(np.sum(np.multiply(layer1_weights[node], features)) + layer1_biases[node])

    #print(layer1_outputs)

    # output calculation at the output layer.

    for node in np.ndindex(outlayer_outputs.shape):
        outlayer_outputs[node] = sigmoid(np.sum(np.multiply(outlayer_weights[node], layer1_outputs)) + outlayer_biases[node])

    #print(outlayer_outputs)

    # error calculation at output layer.

    for node in np.ndindex(outlayer_outputs.shape):
        outlayer_errors[node] = outlayer_outputs[node]*(1-outlayer_outputs[node])*(targets[node]-outlayer_outputs[node])

    print("Error at the output layer : {0} at step {1}".format(outlayer_errors,k))

    # error calculation at hidden layer. 

    for node in np.ndindex(layer1_errors.shape):
        layer1_errors[node] = layer1_outputs[node]*(1-layer1_outputs[node])*np.sum(np.multiply(outlayer_errors,outlayer_weights[0][node]))

    #print(layer1_errors)

    # weight and error corrections.
      
    i = 0
    for node in np.ndindex(layer1_weight_corrections.shape):
        layer1_weight_corrections[node] = learning_rate*layer1_errors[i]*layer1_outputs[i]
        i = (i+1)%2

    #print(layer1_weight_corrections)

    for node in np.ndindex(layer1_bias_corrections.shape):
        layer1_bias_corrections[node] = learning_rate*layer1_errors[node]

    #print(layer1_bias_corrections)

    layer1_weights = np.add(layer1_weights, layer1_weight_corrections)
    layer1_biases = np.add(layer1_biases, layer1_bias_corrections)

    #print(layer1_weights, layer1_biases,end='\n')

    
    i = 0
    for node in np.ndindex(outlayer_weight_corrections.shape):
        outlayer_weight_corrections[node] = learning_rate*outlayer_errors[i]*outlayer_outputs[i]
        i = (i+1)%1

    #print(outlayer_weight_corrections)

    for node in np.ndindex(outlayer_bias_corrections.shape):
        outlayer_bias_corrections[node] = learning_rate*outlayer_errors[node]

    #print(outlayer_bias_corrections)

    outlayer_weights = np.add(outlayer_weights, outlayer_weight_corrections)
    outlayer_biases = np.add(outlayer_biases, outlayer_bias_corrections)

    #print(outlayer_weights, outlayer_biases,end='\n')
    k = k +1

print("Final network paramters:\n")
print("Input features : {0}".format(features))
print("Targets : {0}".format(targets))
print("Layer-1 weights : {0}".format(layer1_weights))
print("Layer-1 biases : {0}".format(layer1_biases))
print("Output layer weights : {0}".format(outlayer_weights))
print("Output layer baises : {0}".format(outlayer_biases))
print("Final error : {0}".format(outlayer_errors))
