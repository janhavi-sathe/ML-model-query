#!/usr/bin/env python
# coding: utf-8

# # Experiments on $SynTab$ Dataset

#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

import numpy as np
import pandas as pd

import joblib  # 用來存模型

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

prediction_done = False

def train_model():
    # Parameters
    n_classes = 5
    n_features = 10
    n_redundant = 2  # Number of redundant features
    n_samples_per_class = 1000
    random_state = 42

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
    y_proba = model.predict_proba(X_test)
    # Calculate the accuracy
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_acc:.4f}")

    # 儲存數據
    np.save("web_model/X_test.npy", X_test)
    np.save("web_model/y_pred.npy", y_pred)

    # **保存模型**
    joblib.dump(model, "logistic_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("🎉 模型已儲存！")
    global prediction_done
    prediction_done = True
    print(f"Prediction done: {prediction_done}")



'''def find_matching_features(target_value, y_pred, X_test):
    """
    根據輸入的 target_value，查找 y_pred 中所有匹配的索引，
    並返回 X_test 中對應位置的特徵值。
    
    :param target_value: 要查找的目標值
    :param y_pred: 預測標籤的數組或列表
    :param X_test: 測試集特徵數據，必須與 y_pred 對應
    :return: 在 X_test 中對應 y_pred 為 target_value 的特徵值
    """
    # 找到 y_pred 中所有匹配 target_value 的索引
    matching_indices = np.where(np.array(y_pred) == target_value)[0]
    
    # 提取 X_test 中對應的特徵值
    matching_features = X_test[matching_indices]
    
    return matching_features

target_value = 1
result = find_matching_features(target_value, y_pred, X_test)
print("符合條件的特徵值:")
print(result)'''