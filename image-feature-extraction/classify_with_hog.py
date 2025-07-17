import numpy as np
from sklearn.svm import SVC


def evaluate_classifier(name, y_true, y_pred):
    """Print detailed evaluation statistics."""
    correct_matches = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    accuracy = correct_matches / total_samples
    
    print(f"\n{name} Classification Results:")
    print(f"Total test samples: {total_samples}")
    print(f"Correct predictions: {correct_matches}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return accuracy

print("Loading HOG features...")
data = np.load("hog_features.npz", allow_pickle=True)
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

# Print feature dimensions
print(f"\nFeature dimensions:")
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

print("Scaling features...")
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

# Scale both train and test using training statistics
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

print("Training SVM classifier...")
# Modified SVM parameters
svm = SVC(kernel="rbf", gamma="scale", C=1.0, random_state=42)
svm.fit(X_train_scaled, y_train)

print("Making predictions...")
y_pred = svm.predict(X_test_scaled)

evaluate_classifier("HOG-SVM", y_test, y_pred)
