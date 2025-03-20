import numpy as np
import pandas as pd

import joblib  # for model saving

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

prediction_done = False

def train_model():
    # Load the dataset
    df = pd.read_csv("web_model/synthetic_dataset.csv")

    # Separate features (X) and target labels (y)
    X = df.drop(columns=['Class']).values
    y = df['Class'].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Standardize the features (important for neural networks)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    from sklearn.linear_model import LogisticRegression

    # Define the Logistic Regression model with a linear decision boundary
    model = LogisticRegression(max_iter=1000, multi_class='ovr', solver='lbfgs', random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # save data
    np.save("web_model/X_test.npy", X_test)
    np.save("web_model/y_pred.npy", y_pred)

    # save model data
    #joblib.dump(model, "logistic_model.pkl")
    #joblib.dump(scaler, "scaler.pkl")

    global prediction_done
    prediction_done = True
    print(f"Prediction done: {prediction_done}")