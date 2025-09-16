import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical

# Step 1: Load and filter the dataset
df = pd.read_csv('ecg_multi_resolution_180.csv')

# Step 2: Remove classes with less than 2 samples (e.g., PVC)
class_counts = df['Label'].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df = df[df['Label'].isin(valid_classes)]

# Step 3: Prepare features and labels
X = df.drop(columns=['Label']).values
y = df['Label'].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Reshape input for CNN/LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, stratify=y_categorical, random_state=42)

# Step 5: 1D CNN model
cnn_model = Sequential([
    Conv1D(64, kernel_size=5, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_data=(X_test, y_test))

# Step 6: Evaluate CNN
cnn_loss, cnn_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
y_pred_cnn = cnn_model.predict(X_test)
y_pred_labels_cnn = np.argmax(y_pred_cnn, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print(f"\n📊 CNN Test Accuracy: {cnn_acc:.4f}")
print("\n📄 CNN Classification Report:")
print(classification_report(y_true_labels, y_pred_labels_cnn, target_names=le.classes_))

# Plot Confusion Matrix
cnn_cm = confusion_matrix(y_true_labels, y_pred_labels_cnn)
ConfusionMatrixDisplay(confusion_matrix=cnn_cm, display_labels=le.classes_).plot(cmap='Blues')
plt.title("1D CNN - Confusion Matrix")
plt.show()

# Step 7: LSTM model
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(64),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_data=(X_test, y_test))

# Evaluate LSTM
lstm_loss, lstm_acc = lstm_model.evaluate(X_test, y_test, verbose=0)
y_pred_lstm = lstm_model.predict(X_test)
y_pred_labels_lstm = np.argmax(y_pred_lstm, axis=1)

print(f"\n📊 LSTM Test Accuracy: {lstm_acc:.4f}")
print("\n📄 LSTM Classification Report:")
print(classification_report(y_true_labels, y_pred_labels_lstm, target_names=le.classes_))

# Plot Confusion Matrix
lstm_cm = confusion_matrix(y_true_labels, y_pred_labels_lstm)
ConfusionMatrixDisplay(confusion_matrix=lstm_cm, display_labels=le.classes_).plot(cmap='Oranges')
plt.title("LSTM - Confusion Matrix")
plt.show()
