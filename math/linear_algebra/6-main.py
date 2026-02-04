#!/usr/bin/env python3

cat_arrays = __import__('6-howdy_partner').cat_arrays

# Test arrays
arr1 = [1, 2, 3, 4, 5]
arr2 = [6, 7, 8]

# Call the function
result = cat_arrays(arr1, arr2)

# Print results
print(result)  # Expected: [1, 2, 3, 4, 5, 6, 7, 8]
print(arr1)    # Original array should remain unchanged: [1, 2, 3, 4, 5]
print(arr2)    # Original array should remain unchanged: [6, 7, 8]

# Additional tests (optional)
arr3 = []
arr4 = [9, 10]
print(cat_arrays(arr3, arr4))  # Expected: [9, 10]
print(cat_arrays(arr4, arr3))  # Expected: [9, 10]
