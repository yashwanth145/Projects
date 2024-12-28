import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Function to plot a digit
def plot_digit(digit_data):
    digit_image = digit_data.reshape(28, 28)
    plt.imshow(digit_image, cmap="binary")
    plt.axis("off")
    plt.show()

# Load MNIST data
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target']

# Convert labels to integers
y = y.astype(np.uint8)

# Split into training and test sets
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Binary classification: Is the digit 5?
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# Train SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# Predict on a sample digit
some_digit = X[0]  # Example digit from the dataset
print(f"Prediction (is 5?): {sgd_clf.predict([some_digit])}")

# One-vs-One Classification
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
print(f"One-vs-One Prediction: {ovo_clf.predict([some_digit])}")

# Random Forest Classifier
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train)
print(f"Random Forest Prediction: {forest_clf.predict([some_digit])}")

# StandardScaler and multi-label classification
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

# Predict multi-label classification for the sample digit
print(f"Multi-label prediction: {knn_clf.predict([some_digit])}")

# Add noise to data
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

# Train KNN classifier on noisy data
knn_clf.fit(X_train_mod, y_train_mod)

# Denoise a test digit
clean_digit = knn_clf.predict([X_test_mod[2]])

# Visualize original, noisy, and cleaned digits
plt.figure(figsize=(8, 4))
plt.subplot(1, 3, 1)
plot_digit(X_test[2])
plt.title("Original")

plt.subplot(1, 3, 2)
plot_digit(X_test_mod[2])
plt.title("With Noise")

plt.subplot(1, 3, 3)
plot_digit(clean_digit)
plt.title("Denoised")
