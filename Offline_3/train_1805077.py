import torchvision.datasets as ds
from torchvision import transforms
# import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

epsilon = 1e-15

def get_train_validation_dataset():
    train_validation_dataset = ds.EMNIST(root='./data', split='letters',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)
    return train_validation_dataset

def get_test_dataset():
    independent_test_dataset = ds.EMNIST(root='./data', split='letters',
                                train=False,
                                transform=transforms.ToTensor())
    return independent_test_dataset

def preprocess_EMNIST_data(dataset):
    data = dataset.data.reshape((-1, 28*28)).float().numpy()
    # Normalization of pixel values
    data /= 255.0
    
    # One-hot encoding for labels
    labels = dataset.targets.numpy()
    num_classes = len(np.unique(labels)) 
    labels = np.eye(num_classes)[labels - 1]
    
    return data, labels

def split_train_validation_data(data, labels, test_ratio=0.15):
    # RANDOM
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=test_ratio, random_state=77)
    return X_train, X_val, y_train, y_val

## Feed Forward Neural Network

# Base Layer class

class BaseLayer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward_propagation(self, input):
        raise NotImplementedError()
    
    def backward_propagation(self, output_gradient, learning_rate):
        raise NotImplementedError()
    
# Dense Layer class

class DenseLayer(BaseLayer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        # RANDOM
        self.weights = np.random.randn(self.input_size, self.output_size)
        self.biases = np.random.randn(self.output_size)
        
    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output 

    def backward_propagation(self, output_gradient, learning_rate):
        input_gradient = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.dot(self.input.T, output_gradient)
        biases_gradient = np.sum(output_gradient, axis=0)
        
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient
        
        return input_gradient

class ActivationLayer(BaseLayer):
    def __init__(self, activation, derivative):
        self.activation = activation
        self.derivative = derivative
        
    def forward_propagation(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output
    
    def backward_propagation(self, output_gradient, learning_rate):
        input_gradient = np.multiply(output_gradient, self.derivative(self.input))
        return input_gradient
    
class ReLU(ActivationLayer):
    def __init__(self):
        super().__init__(self.relu, self.relu_derivative)
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
class DropoutLayer(BaseLayer):
    def __init__(self, dropout_prob):
        self.dropout = dropout_prob
        self.mask = None
        
    def forward_propagation(self, input):
        self.input = input
        if self.dropout == 0:
            self.output = self.input
            return self.output
        elif self.dropout == 1:
            self.output = np.zeros(self.input.shape)
            return self.output
        self.mask = (np.random.rand(*self.input.shape) > self.dropout).astype(float)
        self.output = np.multiply(self.mask, self.input) / (1 - self.dropout)
        return self.output
    
    def backward_propagation(self, output_gradient, learning_rate):
        input_gradient = np.multiply(self.mask, output_gradient)
        return input_gradient
    
class Tanh(ActivationLayer):
    def __init__(self):
        super().__init__(self.tanh, self.tanh_derivative)
        
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.power(np.tanh(x), 2)
    
class Loss:
    def error(self, y_true, y_pred):
        raise NotImplementedError()
    
    def error_derivative(self, y_true, y_pred):
        raise NotImplementedError()
    
class L2Loss(Loss):
    def error(self, y_true, y_pred):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return np.mean(np.power(y_true - y_pred, 2)) 
    
    def error_derivative(self, y_true, y_pred):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return 2 * (y_pred - y_true) / np.size(y_true)
    
class CrossEntropyLoss(Loss):
    def error(self, y_true, y_pred):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.multiply(y_true, np.log(y_pred)))
    
    def error_derivative(self, y_true, y_pred):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.divide(y_true, y_pred) / np.size(y_true)
    
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        
    def forward_propagation(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output
    
    def backward_propagation(self, output_gradient, learning_rate):
        input_gradient = output_gradient
        for layer in reversed(self.layers):
            input_gradient = layer.backward_propagation(input_gradient, learning_rate)
        return input_gradient
    
    def train(self, Loss, X_train, y_train, epochs, learning_rate, batch_size):
        self.loss = Loss
        loss = []
        for i in range(epochs):
            for j in range(0, len(X_train), batch_size):
                X_batch = X_train[j:j+batch_size]
                y_batch = y_train[j:j+batch_size]
                y_pred = self.forward_propagation(X_batch)
                loss.append(self.loss.error(y_batch, y_pred))
                output_gradient = self.loss.error_derivative(y_batch, y_pred)
                self.backward_propagation(output_gradient, learning_rate)
            print("Epoch: ", i, " Loss: ", loss[-1])
        return loss
    
    def predict(self, X):
        y_pred = self.forward_propagation(X)
        return y_pred
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        y = np.argmax(y, axis=1)
        return accuracy_score(y, y_pred), f1_score(y, y_pred, average='macro'), confusion_matrix(y, y_pred)
    
        
        
    


## Main function

if __name__ == "__main__":
    train_validation_dataset = get_train_validation_dataset()
    train_validation_data, train_validation_labels = preprocess_EMNIST_data(train_validation_dataset)
    X_train, X_val, y_train, y_val = split_train_validation_data(train_validation_data, train_validation_labels)
    
    independent_test_dataset = get_test_dataset()
    X_test, y_test = preprocess_EMNIST_data(independent_test_dataset)
    
    # print all shapes
    # print("X_train shape: ", X_train.shape)
    # print("y_train shape: ", y_train.shape)
    # print("X_val shape: ", X_val.shape)
    # print("y_val shape: ", y_val.shape)
    # print("X_test shape: ", X_test.shape)
    # print("y_test shape: ", y_test.shape)
    
    nn = NeuralNetwork([
        DenseLayer(784, 26),
        DropoutLayer(0.2),
        ReLU(),
    ])
    
    nn.train(CrossEntropyLoss(), X_train, y_train, 10, 0.01, 32)
    accuracy_score, f1_score, confusion_matrix = nn.evaluate(X_val, y_val)
    print("Accuracy score: ", accuracy_score)
    print("F1 score: ", f1_score)
    # print("Confusion matrix: ", confusion_matrix)
    
    
    
    