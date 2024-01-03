import torchvision.datasets as ds
from torchvision import transforms
# import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from tqdm import tqdm
import pickle

epsilon = 1e-15
np.random.seed(77)

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

class BaseLayer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward_propagation(self, input):
        raise NotImplementedError()
    
    def backward_propagation(self, output_gradient, learning_rate):
        raise NotImplementedError()
    
def xavier_init(input_size, output_size):
    return np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))

class Optimizer:
    def __init__(self):
        pass
    
    def update(self, weights, biases, weights_gradient, biases_gradient):
        raise NotImplementedError()
        
class MiniBatchGD(Optimizer):
    def __init__(self):
        pass
    
    def update(self, weights, biases, weights_gradient, biases_gradient, learning_rate):
        weights -= learning_rate * weights_gradient
        biases -= learning_rate * biases_gradient        

class AdamOptimizer(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.mw = None
        self.mb = None
        self.vw = None
        self.vb = None
        self.t = 0
        
    def update(self, weights, biases, weights_gradient, biases_gradient, learning_rate):
        self.t += 1
        # print(self.t)
        if self.mw is None:
            self.mw = np.zeros_like(weights)
            self.mb = np.zeros_like(biases)
            self.vw = np.zeros_like(weights)
            self.vb = np.zeros_like(biases)
        self.mw = self.beta1 * self.mw + (1 - self.beta1) * weights_gradient
        self.mb = self.beta1 * self.mb + (1 - self.beta1) * biases_gradient
        self.vw = self.beta2 * self.vw + (1 - self.beta2) * np.power(weights_gradient, 2)
        self.vb = self.beta2 * self.vb + (1 - self.beta2) * np.power(biases_gradient, 2)
        mw_hat = self.mw / (1 - np.power(self.beta1, self.t))
        mb_hat = self.mb / (1 - np.power(self.beta1, self.t))
        vw_hat = self.vw / (1 - np.power(self.beta2, self.t))
        vb_hat = self.vb / (1 - np.power(self.beta2, self.t))
        weights -= learning_rate * mw_hat / (np.sqrt(vw_hat) + self.epsilon)
        biases -= learning_rate * mb_hat / (np.sqrt(vb_hat) + self.epsilon)
        

class DenseLayer(BaseLayer):
    def __init__(self, input_size, output_size, initializer=xavier_init, optimizer=AdamOptimizer):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = initializer(self.input_size, self.output_size)
        self.biases = initializer(1, self.output_size)
        self.optimizer = optimizer()
        
    def forward_propagation(self, input):
        self.input = input
        # print(self.input.shape)
        self.output = np.dot(self.input, self.weights) + self.biases
        # print(self.output.shape)
        return self.output 

    def backward_propagation(self, output_gradient, learning_rate):
        # print(output_gradient.shape, self.weights.shape, self.input.shape, self.biases.shape)
        input_gradient = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.dot(self.input.T, output_gradient)
        biases_gradient = np.sum(output_gradient, axis=0)
        
        # print(self.weights.shape, self.biases.shape)
        # print(weights_gradient.shape, biases_gradient.shape)
        self.optimizer.update(self.weights, self.biases, weights_gradient, biases_gradient, learning_rate)
        
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
    
class Tanh(ActivationLayer):
    def __init__(self):
        super().__init__(self.tanh, self.tanh_derivative)
        
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.power(np.tanh(x), 2)
    
class Sigmoid(ActivationLayer):
    def __init__(self):
        super().__init__(self.sigmoid, self.sigmoid_derivative)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return np.multiply(self.sigmoid(x), 1 - self.sigmoid(x))
    
def softmax(x):
    expo = np.exp(x - np.max(x, axis=1, keepdims=True))
    expsum = np.sum(expo, axis=1, keepdims=True)
    return expo / expsum
    
class Softmax(ActivationLayer):
    def __init__(self):
        super().__init__(softmax, self.return_ones)
        
    def return_ones(self, x):
        return np.ones(x.shape)

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
        self.output = np.multiply(self.input, self.mask) / (1 - self.dropout)
        return self.output
    
    def backward_propagation(self, output_gradient, learning_rate):
        input_gradient = np.multiply(output_gradient, self.mask) / (1 - self.dropout)
        return input_gradient
    
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
        # return -np.divide(y_true, y_pred) / np.size(y_true)
        return y_pred - y_true
    
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
    
    def fit(self, Loss, X_train, y_train, X_val, y_val, epochs, learning_rate, batch_size):
        # open("train_1805077.txt", "w").close()
        self.loss = Loss
        epoch_loss = []
        epoch_accuracy = []
        epoch_val_loss = []
        epoch_val_accuracy = []
        epoch_val_f1 = []
        for epoch in tqdm(range(epochs)):
            train_loss = 0
            for j in range(0, len(X_train), batch_size):
                X_batch = X_train[j:j+batch_size]
                y_batch = y_train[j:j+batch_size]
                y_pred = self.forward_propagation(X_batch)
                
                # print(self.loss.error(y_batch, y_pred))
                train_loss += self.loss.error(y_batch, y_pred)
                # output_gradient = self.loss.error_derivative(y_batch, y_pred)
                # if final layer is softmax
                y_batch_size = y_batch.shape[0]
                if isinstance(self.layers[-1], Softmax):
                    output_gradient = (y_pred - y_batch) / y_batch_size
                output_gradient = self.backward_propagation(output_gradient, learning_rate)
            epoch_loss.append(train_loss / (len(X_train) / batch_size))
            epoch_val_loss.append(self.loss.error(y_val, self.forward_propagation(X_val)))
            temp_accuracy, _ = self.evaluate(X_train, y_train)
            epoch_accuracy.append(temp_accuracy)
            temp_accuracy, temp_f1 = self.evaluate(X_val, y_val)
            epoch_val_accuracy.append(temp_accuracy)
            epoch_val_f1.append(temp_f1)
            # write to file
            # with open("train_1805077.txt", "a") as f:
            #     f.write("Epoch: " + str(epoch+1) + "\n")
            #     f.write("Loss: " + str(epoch_loss[-1]) + " Val loss: " + str(epoch_val_loss[-1]) + " Train acc: " + str(epoch_accuracy[-1]) + " Val acc: " + str(epoch_val_accuracy[-1]) + " Val F1: " + str(epoch_val_f1[-1]) + "\n")
            
            gc.collect() if epoch % 5 == 0 else None
        # print(len(epoch_loss), len(epoch_val_loss), len(epoch_accuracy), len(epoch_val_accuracy), len(epoch_val_f1))    
        return epoch_loss, epoch_val_loss, epoch_accuracy, epoch_val_accuracy, epoch_val_f1
    
    def predict(self, X):
        # predict through the network without Dropout
        pred_nn = NeuralNetwork(self.layers)
        pred_nn.layers = [layer for layer in self.layers if not isinstance(layer, DropoutLayer)]
        y_pred = pred_nn.forward_propagation(X)
        return y_pred
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        y = np.argmax(y, axis=1)
        # print y_pred, y first 10 elements
        # print(y_pred[:10], y[:10])
        return accuracy_score(y, y_pred), f1_score(y, y_pred, average='macro')
    
    def save_confusion_matrix(self, X, y):
        y_pred = self.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        y = np.argmax(y, axis=1)
        matrix = confusion_matrix(y, y_pred)
        df_cm = pd.DataFrame(matrix, index = [i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
                  columns = [i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])
        plt.figure(figsize = (20,14))
        sns.heatmap(df_cm, annot=True, cmap="Blues", fmt='g')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("confusion_matrix_1805077.png")
        plt.clf()
        return confusion_matrix
    
    def save_graphs(self, epoch_loss, epoch_val_loss, epoch_accuracy, epoch_val_accuracy, epoch_val_f1):
        plt.plot(epoch_loss, label="Train loss")
        plt.plot(epoch_val_loss, label="Validation loss")
        plt.legend()
        plt.savefig("loss_1805077.png")
        plt.clf()
        plt.plot(epoch_accuracy, label="Train accuracy")
        plt.plot(epoch_val_accuracy, label="Validation accuracy")
        plt.legend()
        plt.savefig("accuracy_1805077.png")
        plt.clf()
        plt.plot(epoch_val_f1, label="Validation F1 score")
        plt.legend()
        plt.savefig("f1_1805077.png")
        plt.clf()
        return
    
    def clean_model(self):
        # for dense layer keep the weights and biases
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.input = None
                layer.output = None
            elif isinstance(layer, DropoutLayer):
                layer.input = None
                layer.output = None
                layer.mask = None
            else:
                layer.input = None
                layer.output = None
                
    
    def save_model(self):
        self.clean_model()
        with open("model_1805077.pkl", "wb") as f:
            pickle.dump(self, f)


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
        DenseLayer(784, 128),
        ReLU(),
        DropoutLayer(0.5),
        DenseLayer(128, 64),
        ReLU(),
        DenseLayer(64, 26),
        Softmax()
    ])
    
    epoch_loss, epoch_val_loss, epoch_accuracy, epoch_val_accuracy, epoch_val_f1 = nn.fit(CrossEntropyLoss(), 
                                                                                          X_train, 
                                                                                          y_train, 
                                                                                          X_val, 
                                                                                          y_val, 
                                                                                          epochs=20, 
                                                                                          learning_rate=5e-3, 
                                                                                          batch_size=128)
    nn.save_graphs(epoch_loss, epoch_val_loss, epoch_accuracy, epoch_val_accuracy, epoch_val_f1)
    
    accuracy_score, f1_score = nn.evaluate(X_test, y_test)
    nn.save_confusion_matrix(X_test, y_test)
    print("Test Accuracy score: ", accuracy_score)
    print("Test F1 score: ", f1_score)
    
    nn.save_model()
    
    
    
    