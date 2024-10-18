import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
# Load the dataset (replace 'your_dataset.csv' with your actual dataset file)
data = pd.read_csv('D:/Desktop/DWM/diabetes.csv')
# Impute missing values with various techniques
# a. Remove rows/attributes with missing values
data_without_missing = data.dropna()

# b. Replace with mean or mode (you can choose which columns to apply this to)
imputer_mean = SimpleImputer(strategy='mean')
data_mean_imputed = data.copy()
columns_to_impute = ['Pregnancies Glucose ', 'BloodPressure']  # Replace with the columns you want to impute
data_mean_imputed[columns_to_impute] = imputer_mean.fit_transform(data_mean_imputed[columns_to_impute])

# c. Perform data transformation using MinMaxScaler or MaxAbsScaler
scaler = MinMaxScaler()  # You can change this to MaxAbsScaler if needed
columns_to_normalize = ['SkinThickness', 'Insulin']  # Replace with the columns you want to normalize
data_normalized = data.copy()
data_normalized[columns_to_normalize]= scaler.fit_transform(data_normalized[columns_to_normalize])
# Print the first few rows of each imputed/transformed dataset for verification
print("Data without missing values:")
print(data_without_missing.head())
print("\nData with missing values imputed using mean/mode:")
print(data_mean_imputed.head())
print("\nData after normalization:")
print(data_normalized.head())
