import numpy as np
import matplotlib.pyplot as plt


## sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
## sigmoid derivative
def sigmoid_drv(x):
    return sigmoid(x)*(1 - sigmoid(x))
## ReLU function
def relu(x):
    return np.maximum(0, x)
## ReLU derivative
def relu_drv(x):
    return np.where(x > 0, 1, 0)
## tanh function
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
## tanh derivative
def tanh_drv(x):
    return 1 - tanh(x)**2



x = np.linspace(-10, 10, 1000)

# plot
plt.plot(x, sigmoid(x))
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("sigmoid_function.png")
plt.close()

plt.plot(x, sigmoid_drv(x))
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("sigmoid_derivative.png")
plt.close()

plt.plot(x, relu(x))
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("relu_function.png")
plt.close()

plt.plot(x, relu_drv(x))
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("relu_derivative.png")
plt.close()

plt.plot(x, tanh(x))
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("tanh_function.png")
plt.close()

plt.plot(x, tanh_drv(x))
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("tanh_derivative.png")
plt.close()