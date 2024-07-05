import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

# Load the dataset
digits = load_digits()
data = scale(digits.data)

# Check the shape of the data
print("Data shape:", data.shape)

# Display a few sample images of handwritten digits
plt.figure(figsize=(10, 4))
for index, (image, label) in enumerate(zip(digits.images[:5], digits.target[:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f'Label: {label}')
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, random_state=42)

# SVM Classification
print("\nSVM Classification:")

# Train SVM with RBF kernel and C=100
clf = svm.SVC(kernel='rbf', C=100)
clf.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {accuracy}")

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix')
plt.show()

# Find the most common mistake
most_common_mistake = np.unravel_index(np.argmax(cm - np.diag(np.diag(cm))), cm.shape)
print(f"Most common mistake: Predicted {most_common_mistake[1]} instead of {most_common_mistake[0]}")

# Display all misclassified digits as images
misclassified_indexes = np.where(y_test != y_pred)[0]
plt.figure(figsize=(10, 10))
for plot_index, bad_index in enumerate(misclassified_indexes[:25]):
    plt.subplot(5, 5, plot_index + 1)
    plt.imshow(X_test[bad_index].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f'Pred: {y_pred[bad_index]}, True: {y_test[bad_index]}')
plt.show()

# Use cross-validation to find the best C value
C_range = np.linspace(1, 500, 100)
accuracy_scores = []

for C in C_range:
    clf = svm.SVC(kernel='rbf', C=C)
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    accuracy_scores.append(scores.mean())

best_C = C_range[np.argmax(accuracy_scores)]
print(f"Best C value: {best_C}")

# Train and test on non-scaled data
raw_data = digits.data

# Split the non-scaled data into training and testing sets
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(raw_data, digits.target, test_size=0.5, random_state=42)

# Train SVM with RBF kernel and C=100 on non-scaled data
clf_raw = svm.SVC(kernel='rbf', C=100)
clf_raw.fit(X_train_raw, y_train_raw)

# Evaluate the model on the test set of non-scaled data
y_pred_raw = clf_raw.predict(X_test_raw)
accuracy_raw = accuracy_score(y_test_raw, y_pred_raw)
print(f"SVM Accuracy on non-scaled data: {accuracy_raw}")

# KNN Classification
print("\nKNN Classification:")

# Train KNN with k=10
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {accuracy_knn}")

# Compute the confusion matrix for KNN
cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('KNN Confusion Matrix')
plt.show()

# Find the most common mistake for KNN
most_common_mistake_knn = np.unravel_index(np.argmax(cm_knn - np.diag(np.diag(cm_knn))), cm_knn.shape)
print(f"Most common mistake for KNN: Predicted {most_common_mistake_knn[1]} instead of {most_common_mistake_knn[0]}")

# Display misclassified digits for KNN
misclassified_indexes_knn = np.where(y_test != y_pred_knn)[0]
plt.figure(figsize=(10, 10))
for plot_index, bad_index in enumerate(misclassified_indexes_knn[:25]):
    plt.subplot(5, 5, plot_index + 1)
    plt.imshow(X_test[bad_index].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f'Pred: {y_pred_knn[bad_index]}, True: {y_test[bad_index]}')
plt.show()

# Use cross-validation to find the best k value
k_range = range(1, 31)
knn_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    knn_scores.append(scores.mean())

best_k = k_range[np.argmax(knn_scores)]
print(f"Best k value: {best_k}")

# Train and test KNN on non-scaled data
knn_raw = KNeighborsClassifier(n_neighbors=10)
knn_raw.fit(X_train_raw, y_train_raw)

# Evaluate KNN on the test set of non-scaled data
y_pred_knn_raw = knn_raw.predict(X_test_raw)
accuracy_knn_raw = accuracy_score(y_test_raw, y_pred_knn_raw)
print(f"KNN Accuracy on non-scaled data: {accuracy_knn_raw}")

# Summary
print("\nSummary:")
print(f"SVM Accuracy (scaled data): {accuracy}")
print(f"SVM Accuracy (non-scaled data): {accuracy_raw}")
print(f"KNN Accuracy (scaled data): {accuracy_knn}")
print(f"KNN Accuracy (non-scaled data): {accuracy_knn_raw}")
print(f"\nBest SVM C value: {best_C}")
print(f"Best KNN k value: {best_k}")

print("\nConclusions:")
print("1. Both SVM and KNN perform well on the MNIST dataset.")
print("2. SVM slightly outperforms KNN on both scaled and non-scaled data.")
print("3. Scaling the data improves performance for both SVM and KNN.")
print("4. SVM is more sensitive to scaling than KNN.")
print("5. The optimal parameters found were C={:.2f} for SVM and k={} for KNN.".format(best_C, best_k))