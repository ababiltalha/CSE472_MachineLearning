import pandas as pd
import numpy as np


class LogisticRegressionWeakLearning:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y, k=0):
        X = self.feature_selection(X, y, k)
        
        # row means number of samples, column means number of features
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0 #w0
        
        self.gradient_descent(X, y, num_samples, decaying_learning_rate=True)
        
    def gradient_descent(self, X, y, num_samples, early_stopping_threshold=0.5 ,decaying_learning_rate=False):
        learning_rate = self.learning_rate
        for epoch in range(self.num_iterations):
            y_predicted = self.sigmoid(np.dot(X, self.weights) + self.bias)
            
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
            
            loss = self.loss(y_predicted, y)
            if (epoch - 99) % 100 == 0:
                print(f"Loss at epoch {epoch}: {loss}")
                print(learning_rate)
            if loss < early_stopping_threshold:
                print(f"Loss at epoch {epoch}: {loss}")
                break
            
            if decaying_learning_rate:
                # inverse time decay
                learning_rate = self.learning_rate / (1 + epoch * self.learning_rate)
            
    def predict(self, X):
        y_predicted = self.sigmoid(np.dot(X, self.weights) + self.bias)
        y_pred = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_pred)
    
    def loss(self, y_predicted, y_actual):
        # prevent division by 0
        y_predicted = np.clip(y_predicted, 1e-15, 1 - 1e-15)
        return -(1 / len(y_actual)) * np.sum(y_actual * np.log(y_predicted) + (1 - y_actual) * np.log(1 - y_predicted))
    
    def calculate_metrics(self, y_actual, y_pred):
        # calculate accuracy, recall, specificity, precision, false discovery rate, f1 score
        tp = np.sum(np.logical_and(y_pred == 1, y_actual == 1))
        tn = np.sum(np.logical_and(y_pred == 0, y_actual == 0))
        fp = np.sum(np.logical_and(y_pred == 1, y_actual == 0))
        fn = np.sum(np.logical_and(y_pred == 0, y_actual == 1))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp) if tp + fp != 0 else 0
        fdr = fp / (tp + fp) if tp + fp != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
        
        return accuracy, recall, specificity, precision, fdr, f1
    
    def print_metrics(self, y_actual, y_pred):
        accuracy, recall, specificity, precision, fdr, f1 = self.calculate_metrics(y_actual, y_pred)
        print("Accuracy: \t", accuracy)
        print("Recall: \t", recall)
        print("Specificity: \t", specificity)
        print("Precision: \t", precision)
        print("FDR: \t\t", fdr)
        print("F1 Score: \t", f1)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        entropy = 0
        for count in counts:
            entropy += (count / len(y)) * np.log2(count / len(y))
        return -entropy
    
    def calculate_information_gain(self, X, y):
        total_entropy = self.entropy(y)
        
        return 0
        
    def feature_selection(self, X, y, k):
        if k == 0:
            return X
        
        # calculate information gain for each feature
        information_gain = []
        for col in X.columns.values:
            information_gain.append(self.calculate_information_gain(X[col], y))
        
    