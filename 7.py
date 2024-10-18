import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset (replace 'bank_marketing.csv' with the actual file path)
data = pd.read_csv('D:\DWM/bank_marketing.csv')

# a. Explore data and visualize each attribute
# Display basic statistics and info about the dataset
print("Basic statistics of the dataset:")
print(data.describe())

print("\nInfo about the dataset:")
print(data.info())
# Visualize the target variable distribution
sns.countplot(x='duration', data=data)
plt.xlabel('Subscribed to Term Deposit (1/0)')
plt.ylabel('Count')
plt.title('Distribution of Subscribed to Term Deposit')
plt.show()

# b. Preprocess the data and split into training and testing sets
# Encode categorical variables
label_encoder = LabelEncoder()
data['y'] = label_encoder.fit_transform(data['y'])
data_encoded = pd.get_dummies(data, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome'], drop_first=True)
# Split the data into features (X) and target (y)
X = data_encoded.drop(columns=['y'])
y = data_encoded['y']
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# c. Train a Random Forest Classifier and predict the test set results
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy of the model:", accuracy)

# d. Visualize the confusion matrix and compute precision, recall, F-measure, and support
confusion = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(confusion)
classification_rep = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(classification_rep)
# Visualize the confusion matrix
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
