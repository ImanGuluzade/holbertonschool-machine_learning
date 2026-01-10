#!/usr/bin/env python3
"""
Plot a histogram of student scores for Project A.
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Creates a histogram of 50 student grades:
    - X-axis: Grades, bins every 10 units
    - Y-axis: Number of Students
    - Bars outlined in black
    - Title: Project A
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    plt.figure(figsize=(6.4, 4.8))
    bins = np.arange(0, 110, 10)  # Ensure bins go from 0 to 100 inclusive
    plt.hist(student_grades, bins=bins, edgecolor='black')

    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.show()
