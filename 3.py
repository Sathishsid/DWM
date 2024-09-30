import pandas as pd
import matplotlib.pyplot as plt

#a. Loading data from CSV file
data = pd.read_csv('D:\DWM/brain_size.csv')

# b. Compute basic statistics
# Shape of the data
shape = data.shape

# Number of columns
num_columns = len(data.columns)

# Compute mean for numerical columns
mean_values = data.mean(numeric_only=True)  # Only computes for numeric columns

# Print basic statistics
print("Shape of the data:", shape)
print("Number of columns:", num_columns)
print("Mean values for numerical columns:")
print(mean_values)

# c. Splitting data frame on values of categorical variables
# Replace 'Gender' with your actual categorical variable
categorical_variable = 'Gender'
split_data = data.groupby(categorical_variable)

# Display each group
for group_name, group_data in split_data:
    print(f"\nGroup: {group_name}")
    print(group_data)

# d. Visualize each attribute
# Create histograms for numerical attributes
numerical_attributes = data.select_dtypes(include=['int64', 'float64'])
numerical_attributes.hist(figsize=(10, 8))  # Adjust size for better visualization
plt.tight_layout()  # Adjust subplots to fit into figure area.
plt.show()