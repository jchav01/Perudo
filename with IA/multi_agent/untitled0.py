# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 21:01:26 2024

@author: jules
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the data
x = [
    [0, -73, -432, -255, -535, -350], 
    [0, -83, -239, -210, -405, -268], 
    [0, -144, -94, -159, -339, -390], 
    [0, -121, -254, -323, -238, -383], 
    [0, -129, -258, -354, -285, -349], 
    [0, -108, -99, -407, -248, -408], 
    [0, -70, -180, -452, -215, -324], 
    [0, -4, -206, -252, -418, -463], 
    [0, -128, -120, -224, -293, -324], 
    [0, -143, -173, -141, -379, -339]
]

# Create an array for the x-axis
e = np.arange(len(x))

# Plot all curves on the same graph
plt.figure(figsize=(10, 6))

for i in range(6):
    y_values = [row[i] for row in x]
    plt.plot(e, y_values, label=f'Curve {i+1}')

plt.xlabel('e')
plt.ylabel('x')
plt.title('All Curves on the Same Graph')
plt.legend()
plt.grid(True)