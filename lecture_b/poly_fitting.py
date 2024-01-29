import matplotlib.pyplot as plt
import numpy as np

# 2a)

tao = 2 * np.pi
mu, sigma = 0, 0.1

# Create sine
x = np.linspace(0, tao, 100)
y = np.sin(x)

# Create noisy data
ss = np.random.normal(mu, sigma, 15)
psX = np.random.uniform(0, tao, 15)
psY = np.sin(psX) + ss


# Plot
# Create layout
layout = [
    ["A", "A", "D", "D"],
    ["B", "C", "D", "D"],
    ["E", "E", "E", "E"]
]

fig, axd = plt.subplot_mosaic(layout, figsize=(10, 10))

# Plot sine and scatter
axd['A'].plot(x, y, linewidth=2.0)
axd['A'].set(xlim=(0, tao), xticks=np.arange(0, 7),
             ylim=(-2, 2), yticks=np.arange(-1, 1.5, 0.5))
axd['A'].scatter(psX, psY)
axd['A'].set_title('fig. 1')


# 2b)


def fit_poly(xtrain, ytrain, k):
    X = np.array([[x_i ** ex for ex in range(k+1)] for x_i in xtrain])
    W = np.transpose(ytrain) @ X @ np.linalg.inv((np.transpose(X) @ X))
    return W


def poly(x, W):
    k = np.size(W)
    X = np.array([[x_i ** ex for ex in range(k)] for x_i in x])
    return W @ np.transpose(X)


def mse_poly(x, y, W):
    y_hat = poly(x, W)
    MSE = np.mean((y_hat - y) ** 2)
    return MSE


# Calculate 3rd degree polynomial fit:
coeff3 = fit_poly(psX, psY, 3)
polY = poly(x, coeff3)
MSE_text = f"MSE = {mse_poly(psX, psY, coeff3)}"

# Plot poly
axd['A'].plot(x, polY, linewidth=2.0, label=MSE_text)
axd['A'].legend()


# 2c)

# Create sine
x = np.linspace(0, 2 * tao, 100)
y = np.sin(x)

# Create data points
n_train = np.random.normal(mu, sigma, 15)
x_train = np.random.uniform(0, 2 * tao, 15)
y_train = np.sin(x_train) + n_train

n_test = np.random.normal(mu, sigma, 10)
x_test = np.random.uniform(0, 2 * tao, 10)
y_test = np.sin(x_test) + n_test

# Calculate Ws and corresponding MSEs:
ws = np.zeros((15, 16))
MSE = np.zeros(15)
for k in range(1, 16):
    w_temp = fit_poly(x_train, y_train, k)
    MSE[k-1] = mse_poly(x_train, y_train, w_temp)
    ws[(k-1), 0:w_temp.shape[0]] = w_temp

# All fitted polynomials:
X = np.array([[x_i ** ex for ex in range(16)] for x_i in x])
ys = ws @ np.transpose(X)

# Plot MSE:
axd['B'].set_yscale('log')
axd['B'].bar(range(1, 16), MSE, width=1, edgecolor="white", linewidth=0.7)
axd['B'].set_title('fig. 2')

# Plot sine and a suitable polynomial:
axd['C'].set(xlim=(0, 2 * tao), xticks=np.arange(0, 13),
             ylim=(-2, 2), yticks=np.arange(-2, 2.5, 0.5))
axd['C'].plot(x, y, linewidth=2.0)
axd['C'].scatter(x_train, y_train, c='blue', label="train data")
axd['C'].scatter(x_test, y_test, c='red', label="test data")
axd['C'].plot(x, ys[9], linewidth=2.0)
axd['C'].set_title('fig. 3')
axd['C'].legend()


# 2d)

def ridge_fit_poly(x_train, y_train, k, lamb):
    X = np.array([[x_i ** ex for ex in range(k+1)] for x_i in x_train])
    W = np.transpose(y_train) @ X @ np.linalg.inv((np.transpose(X) @ X + (np.identity(k+1) * lamb)))
    return W


# Hyperparameter optimization:
# Small test sample
ks = np.array(list(range(1, 21)))
lambs = 10 ** np.linspace(-5, 0, 20)
MSE_map1 = np.zeros((np.size(ks), np.size(lambs)))
for k_i in range(np.size(ks)):
    for l_i in range(np.size(lambs)):
        w_temp = ridge_fit_poly(x_train, y_train, ks[k_i], lambs[l_i])
        MSE_map1[k_i, l_i] = mse_poly(x_test, y_test, w_temp)

"""
axd['D'].imshow(np.log(MSE_map1))
axd['D'].set_title('fig. 4')
"""

# Big test sample
n_test2 = np.random.normal(mu, sigma, 1000)
x_test2 = np.random.uniform(0, 2 * tao, 1000)
y_test2 = np.sin(x_test2) + n_test2

for k_i in range(np.size(ks)):
    for l_i in range(np.size(lambs)):
        w_temp = ridge_fit_poly(x_train, y_train, ks[k_i], lambs[l_i])
        MSE_map1[k_i, l_i] = mse_poly(x_test2, y_test2, w_temp)

axd['D'].imshow(np.log(MSE_map1))
axd['D'].set_title('fig. 4')


# 2e)

def perform_cv(x, y, k, lamb, folds):
    foldLen = x.shape[0] // folds
    xs = np.array(np.split(x, folds))
    ys = np.array(np.split(y, folds))

    es = np.zeros(folds)
    x_train = np.zeros(x.shape[0] - foldLen)
    y_train = np.zeros(x.shape[0] - foldLen)
    for j in range(folds):
        # Validation set
        x_test = xs[j]
        y_test = ys[j]
        # Training set
        mask = np.ones(folds, dtype=bool)
        mask[[j]] = False
        x_train = np.concatenate(xs[mask, ...])
        y_train = np.concatenate(ys[mask, ...])
        # Calculate MSE
        W = ridge_fit_poly(x_train, y_train, k, lamb)
        es[j] = mse_poly(x_test, y_test, W)
    return np.mean(es)


# Cross validation over a 100 datasets

k_choice = 4
lamb_choice = 10 ** (-2)
divs = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120]

errors = np.zeros((100, len(divs)))
for t in range(100):
    # Generate a dataset:
    n_samp = np.random.normal(mu, sigma, 120)
    x_samp = np.random.uniform(0, 2 * tao, 120)
    y_samp = np.sin(x_samp) + n_samp
    # Do cv for every divisor of 120:
    for d_i in range(len(divs)):
        errors[t, d_i] = perform_cv(x_samp, y_samp, k_choice, lamb_choice, divs[d_i])

means = np.mean(errors, axis=0)
devs = np.std(errors, axis=0)
ob = means + devs
lb = np.maximum(np.zeros(len(divs)), means - devs)

axd['E'].plot(np.array(divs), ob, linewidth=1.0, linestyle='dashed', c='black')
axd['E'].plot(np.array(divs), lb, linewidth=1.0, linestyle='dashed', c='black')
axd['E'].plot(np.array(divs), means, linewidth=1.0, c='black')
axd['E'].set_title('fig. 5')
plt.show()
