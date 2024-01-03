from train_1805077 import *

if __name__ == '__main__':
    with open('model_1805077.pkl', 'rb') as f:
        model = pickle.load(f)
        
    X_test, y_test = preprocess_EMNIST_data(get_test_dataset())
    accuracy_score, f1_score = model.evaluate(X_test, y_test)
    # nn.save_confusion_matrix(X_test, y_test)
    print("Test Accuracy score: ", accuracy_score)
    print("Test F1 score: ", f1_score)