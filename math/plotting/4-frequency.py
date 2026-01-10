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
    
    # Correct histogram
    plt.hist(student_grades, bins=np.arange(0, 110, 10), edgecolor='black')
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.show()
