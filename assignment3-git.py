#%%
from py_compile import main

import git
import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Users\milos\Downloads")
# Load datasets
df1 = pd.read_csv('Trail1.csv')
df2 = pd.read_csv('Trail2.csv')
df3 = pd.read_csv('Trail3.csv')


# Check the shape of each dataset
print(df1.shape)
print(df2.shape)
print(df3.shape)

print(df1.columns)

# Combine into one unified dataset
df_all = pd.concat([df1, df2, df3], ignore_index=True)


cols_to_remove = ['start_time', 'axle', 'cluster', 'tsne_1', 'tsne_2']

df_all['event'] = (df_all['event'] != 'normal').astype(int)

df_all = df_all.drop(columns=cols_to_remove)

print(df_all.shape)
print(df_all.head())

# Normalize the data
# Import StandardScaler for standardizing features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler instance
scaler = StandardScaler()

# Extract all feature columns (exclude the target 'event' column)
features = df_all.drop(columns=['event'])

# Fit the scaler on the features and transform the data to have mean=0 and std=1
features_scaled = scaler.fit_transform(features)

# Convert the scaled numpy array back to a DataFrame with original column names
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

#Grade 4 below: 

#Split the data into training and testing sets in an 80/20 ratio.
from sklearn.model_selection import train_test_split
# Define the target variable and features
X = features_scaled_df
y = df_all['event']
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Perform k-fold cross-validation (e.g., 5-fold) on the training set to evaluate model stability.
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
# Create a Random Forest Classifier instance
rf_classifier = RandomForestClassifier(random_state=42)
# Perform 5-fold cross-validation on the training set
cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5)
# Print the cross-validation scores and their mean
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create SVM model
svm_model = SVC(kernel='rbf', random_state=42)

# Train the model on the training data
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test)

# Calculate accuracy
svm_accuracy = accuracy_score(y_test, y_pred)

print("SVM Accuracy (80/20 split):", svm_accuracy)

from sklearn.model_selection import cross_val_score

# Create SVM model
svm_model = SVC(kernel='rbf', random_state=42)

# Perform 5-fold cross-validation
svm_cv_scores = cross_val_score(svm_model, X, y, cv=5)

print("SVM Cross-validation scores:", svm_cv_scores)
print("Mean CV accuracy:", np.mean(svm_cv_scores))
print("Standard deviation:", np.std(svm_cv_scores))