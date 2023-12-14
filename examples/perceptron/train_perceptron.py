import sys
sys.path.append('../../')

import numpy as np
from models import perceptron as p
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


iris = datasets.load_iris()
X = iris.data[:, :4]  # we only take the first two features for simplicity
y = iris.target

# Convert to binary classification (class 0 vs class 1)
X = X[y < 2]
y = y[y < 2]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature scaling for improved performance
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Create a perceptron instance and train it
perceptron = p.Perceptron(input_dim=4)
perceptron.train(X_train_std, y_train)

# Test the trained perceptron
accuracy = perceptron.test(X_test_std, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")