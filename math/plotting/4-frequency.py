#!/usr/bin/env python3
"""Module to plot a histogram of student scores for Project A."""

import numpy as np
import matplotlib.pyplot as plt

def frequency():
    """Plots a histogram of student scores."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    
    # Plot histogram with bins every 10 units, black edges
    plt.hist(student_grades, bins=np.arange(0, 101, 10), edgecolor='black')
    
    # Set labels and title exactly
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    
    plt.show()
