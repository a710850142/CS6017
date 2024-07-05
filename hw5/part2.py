import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/Users/xxy/Desktop/CS6017/hw5/OnlineNewsPopularity.csv'
data = pd.read_csv(file_path)

# Clean column names by stripping any leading/trailing whitespace
data.columns = data.columns.str.strip()

# Drop unnecessary columns
data = data.drop(columns=['url', 'timedelta'])

# Separate predictor variables and target variable
X = data.drop(columns=['shares']).values
y = data['shares'].values

# Define a binary target variable based on the median number of shares
median_shares = np.median(y)
y_binary = (y > median_shares).astype(int)

# Check the min, median, and max number of shares
print(f"Min shares: {y.min()}")
print(f"Median shares: {median_shares}")
print(f"Max shares: {y.max()}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

# KNN Classification
# Train KNN with k=10
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {accuracy_knn}")

# Use cross-validation to find the best k value
k_range = range(1, 31)
knn_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    knn_scores.append(scores.mean())

best_k = k_range[np.argmax(knn_scores)]
print(f"Best k value: {best_k}")

# Plot KNN cross-validation results
plt.figure(figsize=(10, 6))
plt.plot(k_range, knn_scores, marker='o')
plt.xlabel('Value of k for KNN')
plt.ylabel('Cross-validated Accuracy')
plt.title('KNN Cross-validation Results')
plt.show()

# SVM Classification
# Use a subset of the data for SVM due to computational cost
X_subset = X_train[:5000]
y_subset = y_train[:5000]

C_range = np.logspace(-2, 2, 10)
svm_scores = []

for C in C_range:
    clf = SVC(kernel='rbf', C=C)
    scores = cross_val_score(clf, X_subset, y_subset, cv=5, scoring='accuracy')
    svm_scores.append(scores.mean())

best_C = C_range[np.argmax(svm_scores)]
print(f"Best C value for SVM: {best_C}")

# Plot SVM cross-validation results
plt.figure(figsize=(10, 6))
plt.plot(C_range, svm_scores, marker='o')
plt.xscale('log')
plt.xlabel('Value of C for SVM')
plt.ylabel('Cross-validated Accuracy')
plt.title('SVM Cross-validation Results')
plt.show()

# Decision Tree Classification
# Use RandomizedSearchCV to find the best max_depth and min_samples_split
param_dist = {
    'max_depth': range(1, 21),
    'min_samples_split': range(2, 21)
}

dt_clf = DecisionTreeClassifier()
random_search = RandomizedSearchCV(dt_clf, param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

best_params = random_search.best_params_
print(f"Best parameters for Decision Tree: {best_params}")

# Final evaluation on the test set with the best parameters
dt_clf = DecisionTreeClassifier(**best_params)
dt_clf.fit(X_train, y_train)
y_pred_dt = dt_clf.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt}")

# Summarize findings
print("Summary of findings:")
print(f"Best KNN Accuracy: {accuracy_knn} with k={best_k}")
print(f"Best SVM Accuracy: {max(svm_scores)} with C={best_C}")
print(f"Best Decision Tree Accuracy: {accuracy_dt} with parameters: {best_params}")
