import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

data = load_iris()
X = data.data
y = data.target.reshape(-1, 1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): return x * (1 - x)

np.random.seed(42)
input_size = X.shape[1]     
hidden_size = 6
output_size = 3

W1 = np.random.rand(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.rand(hidden_size, output_size)
b2 = np.zeros((1, output_size))

lr = 0.1
epochs = 1000

for epoch in range(epochs):
    z1 = np.dot(X_train, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    error = y_train - a2
    d2 = error * sigmoid_deriv(a2)
    d1 = np.dot(d2, W2.T) * sigmoid_deriv(a1)

    W2 += lr * np.dot(a1.T, d2)
    b2 += lr * np.sum(d2, axis=0, keepdims=True)
    W1 += lr * np.dot(X_train.T, d1)
    b1 += lr * np.sum(d1, axis=0, keepdims=True)

    if epoch % 100 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch} - Loss: {loss:.4f}")

z1 = np.dot(X_test, W1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, W2) + b2
a2 = sigmoid(z2)
acc = np.mean(np.argmax(a2, axis=1) == np.argmax(y_test, axis=1))
print(f"Acurácia no teste: {acc:.2f}")
