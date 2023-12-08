## Preprocessing Telco Churn Data
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

datadir = "/home/ababil/BUET/4-2/CSE472/Datasets/"
telco = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
credit = "creditcard.csv"
adult = "adult/adult.data"
adult_test = "adult/adult.test"

class LogisticRegressionWeakLearning:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = iterations
        self.weights = None
        self.bias = None
        self.selected_features = []
        
    def fit(self, X, y, k=0, early_stopping_threshold=0.5 ,decaying_learning_rate=False):
        X, self.selected_features = self.feature_selection(X, y, k)
        
        # row means number of samples, column means number of features
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0 #w0
        
        self.gradient_descent(X, y, num_samples, early_stopping_threshold, decaying_learning_rate)
        
    def gradient_descent(self, X, y, num_samples, early_stopping_threshold=0.5 ,decaying_learning_rate=False):
        learning_rate = self.learning_rate
        for epoch in range(self.num_iterations):
            y_predicted = self.sigmoid(np.dot(X, self.weights) + self.bias)
            
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
            
            loss = self.loss(y_predicted, y)
            # if (epoch - 99) % 100 == 0:
                # print(f"Loss at epoch {epoch}: \t{loss}")
                # print(learning_rate)
            if loss < early_stopping_threshold:
                # print(f"Loss at epoch {epoch}: \t{loss}")
                break
            
            if decaying_learning_rate:
                # inverse time decay
                learning_rate = self.learning_rate / (1 + epoch * self.learning_rate)
            
    def predict(self, X):
        # predict on the selected features
        X = X[self.selected_features]
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
        # calculate entropy for binary classification
        if len(y) == 0:
            return 0
        p = np.sum(y) / len(y)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p)) if p != 0 and p != 1 else 0
    
    def calculate_information_gain(self, feature, y, num_bins=5):
        total_entropy = self.entropy(y)
        if feature.unique().size == 2:
            # binary feature
            left = y[feature == feature.unique()[0]]
            right = y[feature == feature.unique()[1]]
            left_entropy = self.entropy(left)
            right_entropy = self.entropy(right)
            remainder = (len(left) / len(y) * left_entropy + len(right) / len(y) * right_entropy)
            return total_entropy - remainder
        else:
            # continuous feature
            # find the min and max values of the feature
            min_value = feature.min()
            max_value = feature.max()
            
            bin_size = (max_value - min_value) / num_bins
            # bin the feature values
            bins = []
            for i in range(num_bins):
                bins.append(min_value + bin_size * i)
            bins.append(max_value)
            
            information_gain = total_entropy
            for i in range(len(bins) - 1):
                in_range = y[(feature >= bins[i]) & (feature < bins[i+1])]
                in_entropy = self.entropy(in_range)
                remainder = len(in_range) / len(y) * in_entropy
                information_gain -= remainder
            return information_gain
        
    def feature_selection(self, X, y, k):
        if k == 0 or k >= len(X.columns.values):
            return X, X.columns.values
        
        # calculate information gain for each feature
        information_gain = []
        for col in X.columns.values:
            information_gain.append(self.calculate_information_gain(X[col], y))
            
        # select the top k features
        selected_features = []
        for i in range(k):
            max_index = np.argmax(information_gain)
            selected_features.append(X.columns.values[max_index])
            information_gain[max_index] = -1
        # print("Selected Features: ", selected_features)
        
        return X[selected_features], selected_features
        
class AdaBoost:
    def __init__(self, num_iterations=10):
        self.num_iterations = num_iterations
        self.hypotheses = []
        self.z = []
        
    def adaBoost(self, X, y, K, k = 0, early_stopping_threshold=0.5 ,decaying_learning_rate=True):
        w = np.ones(len(X)) / len(X)
        
        while True:
            # resample the data
            indices = np.random.choice(len(X), len(X), p=w)
            X = X.iloc[indices]
            y = y.iloc[indices]
                        
            # train a weak learner
            model = LogisticRegressionWeakLearning()
            model.fit(X, y, k, early_stopping_threshold, decaying_learning_rate)
            # self.hypotheses.append(model)
            
            # calculate error
            y_pred = model.predict(X)
            error = np.sum(w * (y_pred != y))
            if error == 0:
                # print("Error is 0")
                error = 1e-15
            if error > 0.5:
                # print("Error is greater than 0.5")
                # self.hypotheses.pop()
                continue
            
            # update weights
            for i in range(len(w)):
                # if the prediction is correct, decrease the weight
                if y_pred[i] == y.iloc[i]:
                    w[i] *= error / (1 - error)
                    
            # normalize weights
            w /= np.sum(w)
            
            # calculate weak learner weight
            weak_learner_weight = np.log((1 - error) / error)
            self.hypotheses.append(model)
            self.z.append(weak_learner_weight)
            if len(self.z) == K:
                break
        # normalize the weights
        self.z /= np.sum(self.z)
        
    def weighted_majority_vote(self, X_test, y_test):
        # print(self.z)
        y_pred = np.zeros(len(X_test))
        for i in range(len(self.hypotheses)):
            X = X_test[self.hypotheses[i].selected_features]
            y_pred += self.z[i] * self.hypotheses[i].predict(X)
        y_pred = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred
    
    def print_accuracy(self, y_actual, y_pred):
        accuracy = np.sum(y_actual == y_pred) / len(y_actual)
        print("Accuracy: ", accuracy)
        

def preprocessAndSplit(dataset):
    # apply preprocessing and split into train and test set
    # applicable for telco churn, credit card fraud, adult 50k data
    if dataset == 'telco':
        df = pd.read_csv(datadir+telco)
        print("Preprocessing Telco Churn Data...")
        
        # change total charges to numeric
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        # if there are NaN values, replace them with 0
        for col in df.columns.values:
            if df[col].dtype == 'int64' or df[col].dtype == 'float64':
                df[col] = df[col].fillna(0)

        # drop customerID
        df.drop(['customerID'], axis=1, inplace=True)

        # change the binary columns to 0 and 1
        for col in df.columns.values:
            if len(df[col].unique()) == 2 and df[col].dtype == 'object':
                if df[col].unique()[0] == 'No':
                    df[col] = df[col].map({df[col].unique()[0]:0, df[col].unique()[1]:1})
                else:
                    df[col] = df[col].map({df[col].unique()[0]:1, df[col].unique()[1]:0})
                    
        # recognize outliers in total charges
        for col in df.columns.values:
            if df[col].dtype == 'float64':
                z = np.abs(stats.zscore(df['TotalCharges']))
                threshold = 3
                outliers = np.where(z > threshold)
                if len(outliers[0]) > 0:
                    df.drop(outliers[0], inplace=True)

        # split into features and labels
        features = df[df.columns.values[:-1]]
        labels = df[df.columns.values[-1]]

        encoded_features = pd.get_dummies(features)

        # split into train and test set
        X_train, X_test, y_train, y_test = train_test_split(encoded_features, labels, test_size=0.2, random_state=77, stratify=labels)   
        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=77)

        # standardize the data
        scaler = StandardScaler()
        # fit transform the columns that are not binary
        for col in X_train.columns.values:
            if len(X_train[col].unique()) > 2:
                X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1,1))
                # X_val[col] = scaler.transform(X_val[col].values.reshape(-1,1))
                X_test[col] = scaler.fit_transform(X_test[col].values.reshape(-1,1))
        
        return X_train, X_test, y_train, y_test
    
    elif dataset == 'credit':
        df = pd.read_csv(datadir+credit)
        print("Preprocessing Credit Card Fraud Data...")
        
        # drop Time
        df.drop(['Time'], axis=1, inplace=True)
        
        df = pd.concat([df[df['Class']==1], df[df['Class']==0].sample(n=20000)]).sample(frac=1)
        # df.info()
        
        # split into features and labels
        features = df[df.columns.values[:-1]]
        labels = df[df.columns.values[-1]]
        
        # split into train and test set
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=77, stratify=labels)   
        
        # standardize the data
        scaler = StandardScaler()
        for col in X_train.columns.values:
            X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1,1))
            X_test[col] = scaler.fit_transform(X_test[col].values.reshape(-1,1))
        
        return X_train, X_test, y_train, y_test
    
    elif dataset == 'adult':
        train_df = pd.read_csv(datadir+adult, header=None)
        test_df = pd.read_csv(datadir+adult_test, header=None, skiprows=1)
        print("Preprocessing Adult Data...")
        
        for df in [ train_df, test_df ]:
            # replace missing values in categorical columns with the mode
            # for col in df.columns.values:
            #     if df[col].dtype == 'object':
            #         df[col] = df[col].replace(' ?', df[col].mode()[0])
            
            # drop rows with missing values
            df.replace(' ?', np.nan, inplace=True)
            df.dropna(inplace=True)
            
            # fix the target column
            df[14] = df[14].map({' <=50K':0, ' >50K':1, ' <=50K.':0, ' >50K.':1})
            
            # change the binary columns to 0 and 1
            for col in df.columns.values:
                if len(df[col].unique()) == 2 and df[col].dtype == 'object':
                    df[col] = df[col].map({df[col].unique()[0]:1, df[col].unique()[1]:0})
                    
        # split into features and labels
        train_features = train_df[train_df.columns.values[:-1]]
        train_labels = train_df[train_df.columns.values[-1]]
        
        test_features = test_df[test_df.columns.values[:-1]]
        test_labels = test_df[test_df.columns.values[-1]]
        
        # one hot encode the categorical columns
        encoded_train_features = pd.get_dummies(train_features)
        encoded_test_features = pd.get_dummies(test_features)
        
        # add the missing columns in the test set
        missing_cols = set(encoded_train_features.columns) - set(encoded_test_features.columns)
        for col in missing_cols:
            encoded_test_features[col] = 0
        encoded_test_features = encoded_test_features[encoded_train_features.columns]
        
        # split into train and test set
        X_train, X_test, y_train, y_test = encoded_train_features, encoded_test_features, train_labels, test_labels
        
        # standardize the data
        scaler = StandardScaler()
        for col in X_train.columns.values:
            if len(X_train[col].unique()) > 2:
                X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1,1))
                X_test[col] = scaler.fit_transform(X_test[col].values.reshape(-1,1))
                
        return X_train, X_test, y_train, y_test
    
def weakLearningStats(datasets):
    for dataset in datasets:
        print("\nDataset: ", dataset)
        X_train, X_test, y_train, y_test = preprocessAndSplit(dataset)
        print("Train Set Size: ", len(X_train))
        print("Test Set Size: ", len(X_test))
        print("Number of Features: ", len(X_train.columns.values))
        print("Number of Positive Labels: ", np.sum(y_train))
        print("Number of Negative Labels: ", len(y_train) - np.sum(y_train))
        print()
    
        model = LogisticRegressionWeakLearning()
        model.fit(
            X_train, 
            y_train, 
            k = 30, 
            early_stopping_threshold=0.5, 
            decaying_learning_rate=True
            )
        y_pred_train = model.predict(X_train)
        print("Training Set Metrics:")
        model.print_metrics(y_train, y_pred_train)
        print("Test Set Metrics:")
        y_pred = model.predict(X_test)
        model.print_metrics(y_test, y_pred)
        print()
        
def adaBoostStats(datasets):
    for dataset in datasets:
        print("\nDataset: ", dataset)
        X_train, X_test, y_train, y_test = preprocessAndSplit(dataset)
        
        for K in [ 5, 10, 15, 20 ]:
            adaBoost = AdaBoost()
            adaBoost.adaBoost(
                X_train, 
                y_train, 
                K, 
                k=30, 
                early_stopping_threshold=0.5, 
                decaying_learning_rate=True
                )
            y_pred_train = adaBoost.weighted_majority_vote(X_train, y_train)
            print("Training Set Metrics:")
            adaBoost.print_accuracy(y_train, y_pred_train)
            print("Test Set Metrics:")
            y_pred = adaBoost.weighted_majority_vote(X_test, y_test)
            adaBoost.print_accuracy(y_test, y_pred)
            print()

def main():
    datasets = [ 'telco', 'credit', 'adult' ]
    # weakLearningStats(datasets)
    adaBoostStats(datasets)
    
if __name__ == "__main__":
    main()