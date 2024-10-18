import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
data = pd.read_csv(url)

# Describe the data
print("Data Description:\n", data.describe())

# Identify missing data
print("\nMissing Data:\n", data.isnull().sum())

# Identify outliers using Z-score
z_scores = np.abs(stats.zscore(data))
outliers = (z_scores > 3).sum(axis=0)
print("\nOutliers:\n", outliers)

# Find correlation among attributes
correlation_matrix = data.corr()
print("\nCorrelation Matrix:\n", correlation_matrix)

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
