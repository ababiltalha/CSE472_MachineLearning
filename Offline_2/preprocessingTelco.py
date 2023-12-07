## Preprocessing Telco Churn Data
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from logisticRegression import LogisticRegressionWeakLearning as lr

datadir = "/home/ababil/BUET/4-2/CSE472/Datasets/"
filename = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

def preprocessAndSplit(df):
    # df.info()

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
        # if object type and only 2 unique values
        if len(df[col].unique()) == 2 and df[col].dtype == 'object':
            # if the first value is 'No' then change it to 0 and the second value to 1
            if df[col].unique()[0] == 'No':
                df[col] = df[col].map({df[col].unique()[0]:0, df[col].unique()[1]:1})
            else:
                df[col] = df[col].map({df[col].unique()[0]:1, df[col].unique()[1]:0})
                
    # recognize outliers in monthly charges
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

    # split into train, validation and test set
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

def main():
    df = pd.read_csv(datadir+filename)
    X_train, X_test, y_train, y_test = preprocessAndSplit(df)
    
    # train a model on the training set and evaluate on the validation set
    model = lr()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print(y_pred)
    model.print_metrics(y_test, y_pred)
    
    
if __name__ == "__main__":
    main()