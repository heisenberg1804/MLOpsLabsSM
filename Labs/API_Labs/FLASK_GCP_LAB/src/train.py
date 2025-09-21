import joblib
import os
import pandas as pd 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def run_training(): 
    """
    Train the model using Support Vector Machine Classifier
    """
    # Read the training data 
    dataset = pd.read_csv('data/IRIS.csv')

    # Split into features and target
    X = dataset.drop("species", axis=1).copy()
    y = dataset["species"].copy()

    # Store feature names before splitting
    feature_names = list(X.columns)

    # Create train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=26)

    # Initialize and train the SVM model
    model = SVC(kernel='rbf', random_state=26, probability=True)
    
    # Create a dictionary to store both model and feature names
    model_info = {
        'model': model,
        'feature_names': feature_names
    }

    # Fit the model
    model.fit(X_train, y_train)

    # Calculate and print accuracy
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")

    # Persist the model info
    if not os.path.exists("../model"):
        os.makedirs("../model")
    joblib.dump(model_info, "../model/model_svm.pkl")

if __name__ == "__main__":
    run_training()