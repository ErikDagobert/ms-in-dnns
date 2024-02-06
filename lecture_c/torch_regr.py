import numpy as np
import torch
# import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

N_TRAIN = 15
SIGMA_NOISE = 0.1

torch.manual_seed(0xDEADBEEF)
x_train = torch.rand(N_TRAIN) * 2 * torch.pi
y_train = torch.sin(x_train) + torch.randn(N_TRAIN) * SIGMA_NOISE


# 1a)

# Exact:
def fit_poly(xtrain, ytrain):
    X = torch.vander(xtrain, N=4, increasing=True)
    a = torch.matmul(ytrain, X)
    b = torch.linalg.inv(torch.matmul(torch.transpose(X, 0, 1), X))
    W = torch.matmul(a, b)
    return W


# Torch:
class PolyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)

    def forward(self, x):
        y = self.linear(x)
        return y


# Initiating model:
model_a = PolyModule()
loss_func = nn.MSELoss()
model_a.linear.weight.data.fill_(1)

# Print model info:
print(f"Model structure: {model_a}\n\n")

for name, param in model_a.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:4]} \n")

# Run model:
x = torch.vander(x_train, N=4, increasing=True)
preds = model_a(x)
targets = torch.sin(x_train)
targets = torch.reshape(targets, (15, 1))
loss = loss_func(preds, targets)
loss.backward()

# Optimize:

# Find a good learning rate
lr_losses = []
for lr in [0.1 ** k for k in np.linspace(4.1, 4.5, 15)]:
    model_a.linear.weight.data.fill_(1)
    sgd = torch.optim.SGD(model_a.parameters(), lr=lr)
    for step in range(100):
        # zero the parameter gradients
        sgd.zero_grad()

        # forward + backward + optimize
        outputs = model_a(x)
        loss = loss_func(outputs, targets)
        loss.backward()
        sgd.step()

    lr_losses.append(loss.item())

losses = np.array(lr_losses)
lr = 0.1 ** np.linspace(4.1, 4.5, 15)[np.argmin(losses)]
print('Best learn rate %f' % lr)

# Repeat with optimal lr
loss_step = torch.zeros(100)

model_a.linear.weight.data.fill_(1)
sgd = torch.optim.SGD(model_a.parameters(), lr=lr)
for step in range(100):
    # zero the parameter gradients
    sgd.zero_grad()

    # forward + backward + optimize
    outputs = model_a(x)
    loss = loss_step[step] = loss_func(outputs, targets)
    loss.backward()
    sgd.step()

# Plot
layout = [
    ["A", "A", "B", "B"],
    ["C", "C", "D", "D"],
    ["E", "E", "F", "F"]
]

fig, axd = plt.subplot_mosaic(layout, figsize=(9, 9))
axd['A'].plot(range(100), loss_step.detach().numpy(), linewidth=2.0)
axd['A'].set_title('fig. 1a')
axd['A'].set_xlabel("Steps")
axd['A'].set_ylabel("MSE loss")

# Ground truth and training data
x_axis = np.linspace(0, 2 * torch.pi, 100)
y_truth = np.sin(x_axis)
axd['B'].plot(x_axis, y_truth, label="Ground truth")
axd['B'].scatter(x_train, y_train, label="Training data")

# Exact solution:
X = torch.tensor([x_axis ** exp for exp in range(4)], dtype=torch.float32)
w_exact = fit_poly(x_train, y_train)
axd['B'].plot(torch.tensor(x_axis), torch.matmul(w_exact, X).flatten(), label="Exact solution")

# SGD solution:
x_axist = torch.linspace(0, 2 * torch.pi, 100)
Xt = torch.vander(x_axist, N=4, increasing=True)
y_sgd = model_a(Xt)
axd['B'].plot(x_axist, y_sgd.detach(), label="Optimized polynomial")

axd['B'].legend()
axd['B'].set_title('fig. 1b')


# 1b)

model_b = PolyModule()
with torch.no_grad():
    model_b.linear.weight = nn.Parameter(torch.tensor([0.1 ** k for k in range(4)]))

# Search for good hyperparameters with new initialization:
loss_step = torch.zeros(100)

# Test with old lr but with new initialization and momentum = 0.9:
model_b = PolyModule()
with torch.no_grad():
    model_b.linear.weight = nn.Parameter(torch.tensor([0.1 ** k for k in range(4)]))
sgd_b = torch.optim.SGD(model_b.parameters(), momentum=0.95, lr=lr)
for step in range(100):
    # zero the parameter gradients
    sgd_b.zero_grad()

    # forward + backward + optimize
    outputs = model_b(x)
    loss = loss_step[step] = loss_func(outputs, targets)
    loss.backward()
    sgd_b.step()
print(loss)

# Plot:
axd['C'].plot(range(100), loss_step.detach().numpy(), linewidth=2.0)
axd['C'].set_title("fig. 2a")
axd['C'].set_xlabel("Steps")
axd['C'].set_ylabel("MSE loss")

# Ground truth and training data
x_axis = np.linspace(0, 2 * torch.pi, 100)
y_truth = np.sin(x_axis)
axd['D'].plot(x_axis, y_truth, label="Ground truth")
axd['D'].scatter(x_train, y_train, label="Training data")

# Fitted polynomial
y_sgd_m = model_b(Xt)
axd['D'].plot(x_axist, y_sgd_m.detach(), label="New optimized polynomial")
axd['D'].legend()
axd['D'].set_title("fig. 2b")


# Adam:
for lr_adam in [0.03666]:
    model_adam = PolyModule()
    with torch.no_grad():
        model_adam.linear.weight = nn.Parameter(torch.tensor([0.1 ** k for k in range(4)]))
    sgd_adam = torch.optim.Adam(model_adam.parameters(), lr=lr_adam)
    for step in range(100):
        # zero the parameter gradients
        sgd_adam.zero_grad()

        # forward + backward + optimize
        outputs = model_adam(x)
        loss = loss_step[step] = loss_func(outputs, targets)
        loss.backward()
        sgd_adam.step()
    print((lr_adam, loss))

# Plot
axd['E'].plot(range(100), loss_step.detach().numpy(), linewidth=2.0)
axd['E'].set_title("fig. 3a")
axd['E'].set_xlabel("Steps")
axd['E'].set_ylabel("MSE loss")

# Ground truth and training data
x_axis = np.linspace(0, 2 * torch.pi, 100)
y_truth = np.sin(x_axis)
axd['F'].plot(x_axis, y_truth, label="Ground truth")
axd['F'].scatter(x_train, y_train, label="Training data")

# Fitted polynomial
y_sgd_adam = model_adam(Xt)
axd['F'].plot(x_axist, y_sgd_adam.detach(), label="Adam solution")
axd['F'].legend()
axd['F'].set_title("fig. 3b")

plt.show()

# LBFGS, ran out of time
