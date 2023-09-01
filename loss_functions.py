import numpy as np

# Mean Squared Error (MSE) Loss
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Cross-Entropy Loss (Binary Cross-Entropy)
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Avoid division by zero
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Hinge Loss
def hinge_loss(y_true, y_pred):
    return np.mean(np.maximum(0, 1 - y_true * y_pred))

# Huber Loss
def huber_loss(y_true, y_pred, delta=1.0):
    absolute_error = np.abs(y_true - y_pred)
    quadratic_loss = 0.5 * (absolute_error ** 2)
    linear_loss = delta * (absolute_error - 0.5 * delta)
    return np.mean(np.where(absolute_error <= delta, quadratic_loss, linear_loss))

# Kullback-Leibler Divergence (KL Divergence)
def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))
