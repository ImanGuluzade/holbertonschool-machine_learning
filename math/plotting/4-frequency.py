#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def frequency():
    """
    Plot a histogram of student grades for Project A
    with bins every 10 units, black edges, and proper labels.
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    
    # 1. Define bins from 0 to 100 in steps of 10
    bins = np.arange(0, 101, 10)
    
    # 2. Plot the histogram with black edges
    plt.hist(student_grades, bins=bins, edgecolor='black')
    
    # 3. Set labels and title
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    
    # 4. Set the exact axis limits to match the reference image
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    
    # 5. Set the x-ticks to be every 10 units
    plt.xticks(bins)
    
    plt.show()
