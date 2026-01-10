#!/usr/bin/env python3
"""
Plot a histogram of student grades for Project A.
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots histogram of student grades:
    - X-axis: Grades
    - Y-axis: Number of Students
    - Bins every 10 units from 0 to 100
    - Bars outlined in black
    - Title: Project A
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    plt.figure(figsize=(6.4, 4.8))

    # The critical part: exact bins and edge color
    bins = np.arange(0, 101, 10)  # bins every 10 units from 0 to 100
    plt.hist(student_grades, bins=bins, edgecolor='black', linewidth=1.2)

    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.show()
