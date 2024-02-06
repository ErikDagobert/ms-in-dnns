import numpy as np
# import matplotlib.pyplot as plt


# 2a)

class NPLinear():
    def __init__(self, n_in, n_out):
        k = np.sqrt(1 / n_in)
        self.x = np.array([])
        self.W = np.random.uniform(-k, k, (n_in, n_out))
        self.b = np.random.uniform(-k, k, (n_out))
        self.W_grad = np.zeros((n_in, n_out))
        self.b_grad = np.zeros(n_out)

    def forward(self, inputs):
        self.x = inputs
        return np.transpose(self.W @ np.transpose(inputs)) + self.b[None, :]

    def backward(self, loss_grad):
        self.W_grad = np.transpose(np.transpose(self.x) @ loss_grad)  # TODO: Check
        self.b_grad = loss_grad.sum(axis=0)
        return loss_grad @ self.W

    def gd_update(self, lr):
        self.W = self.W - lr * self.W_grad
        self.b = self.b - lr * self.b_grad


class NPMSELoss():
    def __init__(self):
        self.preds = np.array([])
        self.targets = np.array([])
        self.MSE = np.array([])
        self.MSE_grad = np.array([])

    def forward(self, preds, targets):
        self.preds = preds
        self.targets = targets
        self.MSE = ((preds - targets) ** 2).mean(axis=1).mean()
        return self.MSE

    def backward(self):
        return (2 / self.preds.size) * (self.preds - self.targets)


class NPReLU():
    def __init__(self):
        self.x = np.array([])
        self.output = np.array([])
        self.grad = np.array([])

    def forward(self, inputs):
        self.x = inputs
        self.output = np.maximum(self.x, np.zeros_like(self.x))
        return self.output

    def backward(self, loss_grad):
        inputs = np.copy(self.output)
        inputs[inputs != 0] = 1
        self.grad = np.multiply(inputs, loss_grad)
        return self.grad


# 2b)


class NPModel():
    def __init__(self):
        self.W1_grad = np.array([])
        self.W2_grad = np.array([])

        self.linear1 = NPLinear(1, 5)
        self.relu = NPReLU()
        self.linear2 = NPLinear(5, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def backward(self, loss_grad):
        pass

    def gd_update(self, lr):
        pass


N_TRAIN = 100
N_TEST = 1000
SIGMA_NOISE = 0.1

np.random.seed(0xDEADBEEF)
x_train = np.random.uniform(low=-np.pi, high=np.pi, size=N_TRAIN)[:, None]
y_train = np.sin(x_train) + np.random.randn(N_TRAIN, 1) * SIGMA_NOISE

x_test = np.random.uniform(low=-np.pi, high=np.pi, size=N_TEST)[:, None]
y_test = np.sin(x_test) + np.random.randn(N_TEST, 1) * SIGMA_NOISE

# for epoch in range(10):
#    print()
