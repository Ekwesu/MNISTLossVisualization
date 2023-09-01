# visualization.py

import numpy as np
import matplotlib.pyplot as plt

def visualize_loss_function(loss_function, function_name):
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.linspace(0.01, 0.99, num=100)  # Predicted values between 0 and 1

    # Calculate loss values using the selected loss function
    loss_values = [loss_function(y_true, y) for y in y_pred]

    # Plot the loss function
    plt.figure(figsize=(8, 6))
    plt.plot(y_pred, loss_values, label=function_name)
    plt.xlabel('Predicted Values')
    plt.ylabel('Loss')
    plt.title(f'{function_name} Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    pass
