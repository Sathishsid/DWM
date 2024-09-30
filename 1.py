import numpy as np
# a. Create multidimensional arrays and find its shape and dimension
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Multidimensional Array:")
print(arr)
print("Shape of the array:", arr.shape)
print("Number of dimensions:", arr.ndim)

# b. Create a matrix full of zeros and ones
zeros_matrix = np.zeros((3, 3))
ones_matrix = np.ones((3, 3))
print("\nMatrix of Zeros:")
print(zeros_matrix)
print("\nMatrix of Ones:")
print(ones_matrix)


# c. Reshape and flatten data in the array
reshaped_arr = arr.reshape(1, 9)
flattened_arr = arr.flatten()
print("\nReshaped Array:")
print(reshaped_arr)
print("\nFlattened Array:")
print(flattened_arr)

# d. Append data vertically and horizontally
arr1 = np.array([[10, 11, 12]])
arr2 = np.array([[13, 14, 15]])
vertical_concatenation = np.vstack((arr1, arr2))
horizontal_concatenation = np.hstack((arr1, arr2))
print("\nVertical Concatenation:")
print(vertical_concatenation)
print("\nHorizontal Concatenation:")
print(horizontal_concatenation)

# e. Apply indexing and slicing on array
print("\nIndexing:")
print(arr[1, 2])  # Access element at row 1, column 2
print("\nSlicing:")
print(arr[0:2, 1:3])  # Slicing rows 0 to 1 and columns 1 to 2
# f. Use statistical functions on array
print("\nMinimum Value:", np.min(arr))
print("Maximum Value:", np.max(arr))
print("Mean Value:", np.mean(arr))
print("Median Value:", np.median(arr))
print("Standard Deviation:", np.std(arr))
