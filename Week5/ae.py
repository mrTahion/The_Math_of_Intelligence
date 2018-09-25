import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Data file path
data_file_path= 'input/glass.csv'

# reading data csv
data = pd.read_csv(data_file_path)


columns_drop= ['Type']
# pull data into type (y) and features (X)
y_train = data.Type
X_train = data.drop(columns_drop, axis=1)

# data prep for PCA
maxs = np.amax(X_train, axis=0)
mins = np.amin(X_train, axis=0)

AE_data = (X_train - mins) / (maxs - mins)
AE_data = AE_data.transpose()

# Sigmoid
def sigmoid(x):
    sigmoid = 1 / (1 + np.exp(-x))
    sigmoid_grad = sigmoid * (1 - sigmoid)
    return sigmoid, sigmoid_grad

# ReLU
def relu(x):
    relu = np.maximum(x, np.zeros(x.shape))
    relu_grad = np.ones(x.shape) * (x > 0)
    return relu, relu_grad


# Make a new layer
def new_layer(dim_in, dim_out):
    weights = np.random.rand(dim_out, dim_in) - 0.5
    bias = np.random.rand(dim_out, 1) - 0.5
    
    return {
        'weights': weights,
        'bias': bias,
        'activations': np.zeros((dim_out, AE_data.shape[1])),
        'act_grad': np.zeros((dim_out, AE_data.shape[1])),
        'errors': np.zeros((dim_out, AE_data.shape[1])),
        'weights_grad': np.zeros(weights.shape),
        'bias_grad': np.zeros(bias.shape)
    }

# Our specific models hyperparameters
def initialize_model():
    return [new_layer(9, 8), new_layer(8, 7),
     new_layer(7, 6), new_layer(6, 5), new_layer(5, 4),
     new_layer(4, 3), new_layer(3, 2), new_layer(2, 3),
     new_layer(3, 4), new_layer(4, 5), new_layer(5, 6),
     new_layer(6, 7), new_layer(7, 8), new_layer(8, 9)]


def forward_propagate(layers, inputs, activation_function):
    for layer in layers:
        zs = np.matmul(layer['weights'], inputs) + layer['bias']
        activations, act_grad = activation_function(zs)
        layer['activations'] = activations
        layer['act_grad'] = act_grad
        inputs = activations

def backward_propagate(layers, inputs, outputs):    
    # Initiate errors for the last layer (normalized)
    errors = (layers[-1]['activations'] - outputs)
    
    # Go backwards through the layers
    for layer_number, layer in reversed(list(enumerate(layers))):
        layer['errors'] = errors
        
        # Calculate Hadamard product between errors and activation gradients
        hadamard = errors * layer['act_grad']
        
        # Get activations of previous layer - if at the first layer use input
        if layer_number == 0:
            last_activations = inputs
        else:
            last_activations = layers[layer_number - 1]['activations']
        
        # Calculate derivatives of weights
        weights_grad = np.matmul(hadamard, last_activations.transpose()) / inputs.shape[1]
        layer['weights_grad'] = weights_grad
        
        # Calculate derivatives of biases
        bias_grad = hadamard.sum(axis=1).reshape(layer['bias'].shape) / inputs.shape[1]
        layer['bias_grad'] = bias_grad
        
        # Backpropagate errors, unless we're at the first layer
        if layer_number != 0:
            errors = np.matmul(layer['weights'].transpose(), hadamard)

def gradient_descent(layers, learning_rate):
    for layer in layers:
        layer['weights'] = layer['weights'] - layer['weights_grad'] * learning_rate
        layer['bias'] = layer['bias'] - layer['bias_grad'] * learning_rate

def error_function(layers):
    errors = layers[-1]['errors']
    e_squared = errors ** 2
    return e_squared.sum() / errors.shape[1]

def trainAE(activation_function, learning_rate, epochs):
    # Initialize model
    layers = initialize_model()
    
    # Array of errors
    error_history = []
    
    # Run through the epochs
    for epoch in range(epochs):
        forward_propagate(layers, AE_data, activation_function)
        backward_propagate(layers, AE_data, AE_data)
        gradient_descent(layers, learning_rate)
        error_history.append(error_function(layers))
    
    # Return trained model, and error_history for plotting
    return layers, error_history

layers, error_history = trainAE(sigmoid, 1, 1000)

# Get encoded coordinates
ec1 = np.array(layers[6]['activations'][0, :].transpose())
ec2 = np.array(layers[6]['activations'][1, :].transpose())

# Plot encoded data
plt.scatter(ec1, ec2, c=y_train)
plt.show()                                