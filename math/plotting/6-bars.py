#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def bars():
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    # Names of people
    people = ['Farrah', 'Fred', 'Felicia']
    
    # Colors for fruits
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']  # apples, bananas, oranges, peaches
    
    # Plot each fruit stacked
    bottom = np.zeros(3)  # starting point for stacking
    for i in range(4):
        plt.bar(people, fruit[i], bottom=bottom, color=colors[i], width=0.5, label=['Apples', 'Bananas', 'Oranges', 'Peaches'][i])
        bottom += fruit[i]  # increment bottom for next stack
    
    # Labels, title, y-axis
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    
    # Add legend
    plt.legend()
    
    # Show plot
    plt.show()
