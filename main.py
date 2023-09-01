import numpy as np
import matplotlib.pyplot as plt
from loss_functions import mean_squared_error, cross_entropy_loss, hinge_loss, huber_loss, kl_divergence
from visualization import visualize_loss_function

if __name__ == '__main__':
    print("Select a loss function to visualize:")
    print("1. Mean Squared Error (MSE)")
    print("2. Cross-Entropy Loss")
    print("3. Hinge Loss")
    print("4. Huber Loss")
    print("5. Kullback-Leibler Divergence (KL Divergence)")
    
    choice = int(input("Enter the number (1-5) of the loss function you want to visualize: "))

    if choice == 1:
        visualize_loss_function(mean_squared_error, "Mean Squared Error (MSE)")
    elif choice == 2:
        visualize_loss_function(cross_entropy_loss, "Cross-Entropy Loss")
    elif choice == 3:
        visualize_loss_function(hinge_loss, "Hinge Loss")
    elif choice == 4:
        visualize_loss_function(huber_loss, "Huber Loss")
    elif choice == 5:
        visualize_loss_function(kl_divergence, "Kullback-Leibler Divergence (KL Divergence)")
    else:
        print("Invalid choice. Please select a number between 1 and 5.")