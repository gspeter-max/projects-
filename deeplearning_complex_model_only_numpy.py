import numpy as np 

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

input_size = 2 
hiden_size_1 = 4 
hiden_size_2 = 4 
output_size = 1 
learning_rate =0.1 
epochs = 100000 

np.random.seed(42)
w1 = np.random.randn(input_size , hiden_size_1)
b1 = np.zeros((1, hiden_size_1))
w2 = np.random.randn(hiden_size_1, hiden_size_2)
b2 = np.zeros((1 , hiden_size_2))
w3 = np.random.randn(hiden_size_2, output_size)
b3 = np.zeros((1, output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # Corrected

def sigmoid_derivative(x):
	return x * (1 - x)

def relu(x):
	return np.maximum(0, x)

def relu_derivative(x):
	return np.where(x > 0, 1, 0)

for epoch in range(epochs): 

    z1 = np.dot(x , w1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1 , w2) + b2 
    a2 = relu(z2)

    z3 = np.dot(a2 , w3) + b3
    a3 = sigmoid(z3)

    loss = -np.mean(y * np.log(a3) + (1 - y) * np.log(1 - a3))

    dL_da3 = a3 - y

    dL_dz3 = dL_da3 * sigmoid_derivative(a3)  # (dL/dz3) = (dL/da3) * (da3/dz3)
    dL_dW3 = np.dot(a2.T, dL_dz3)  # (dL/dW3) = (dL/dz3) * (dz3/dW3)
    dL_db3 = np.sum(dL_dz3, axis=0, keepdims=True)  # (dL/db3)

    # Hidden Layer 2 (Layer 2) - Applying Chain Rule
    dL_da2 = np.dot(dL_dz3, w3.T)  # (dL/da2) = (dL/dz3) * (dz3/da2)
    dL_dz2 = dL_da2 * relu_derivative(a2)  # (dL/dz2) = (dL/da2) * (da2/dz2)
    dL_dW2 = np.dot(a1.T, dL_dz2)  # (dL/dW2) = (dL/dz2) * (dz2/dW2)
    dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)  # (dL/db2)

    # Hidden Layer 1 (Layer 1) - Applying Chain Rule
    dL_da1 = np.dot(dL_dz2, w2.T)  # (dL/da1) = (dL/dz2) * (dz2/da1)
    dL_dz1 = dL_da1 * relu_derivative(a1)  # (dL/dz1) = (dL/da1) * (da1/dz1)
    dL_dW1 = np.dot(x.T, dL_dz1)  # (dL/dW1) = (dL/dz1) * (dz1/dW1)
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True) 


    w3 -= learning_rate * dL_dW3
    w2 -= learning_rate * dL_dW2
    w1 -= learning_rate * dL_dW1
    b3 -= learning_rate * dL_db3
    b2 -= learning_rate * dL_db2
    b1 -= learning_rate * dL_db1

    if epoch % 1000 == 0 : 
        print(f"epoch : {epoch} ----------->loss : {loss:.6f}")


print(f"final prediction ")
prediction = np.round(a3)
print(prediction)
