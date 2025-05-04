"""
Emotion Recognition Model Training Script

This script is used to train a model to recognize emotions based on facial and hand landmarks.
It's based on the original data_training.py but optimized for emotion detection.

Instructions:
1. First collect data using collect_emotion_data.py for various emotions
2. Run this script in the directory containing your .npy data files
3. The trained model will be saved as "model.h5" and labels as "labels.npy"
"""

import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout
from keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set working directory to the location of data files
os.chdir(r"C:\Users\srini\Downloads")
print(f"Working directory: {os.getcwd()}")

# Initialize variables
is_init = False
size = -1
label = []
dictionary = {}
c = 0

print("Looking for emotion data files (.npy)...")

# Load and process data files
for i in os.listdir():
    if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):
        print(f"Found data file: {i}")
        
        if not(is_init):
            is_init = True
            X = np.load(i)
            size = X.shape[0]
            y = np.array([i.split('.')[0]]*size).reshape(-1,1)
        else:
            X = np.concatenate((X, np.load(i)))
            y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1)))

        # Store label information
        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c
        c = c+1

print(f"Found {len(label)} emotion classes: {', '.join(label)}")

# Convert string labels to numeric indices
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

# Convert to one-hot encoding
y = to_categorical(y)

# Shuffle data
print("Shuffling data...")
X_new = X.copy()
y_new = y.copy()
counter = 0

cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt:
    X_new[counter] = X[i]
    y_new[counter] = y[i]
    counter = counter + 1

# Split data into training and validation sets
split_ratio = 0.8
split_idx = int(X_new.shape[0] * split_ratio)
X_train, X_val = X_new[:split_idx], X_new[split_idx:]
y_train, y_val = y_new[:split_idx], y_new[split_idx:]

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, {y_val.shape}")

# Build model with dropout for better generalization
print("Building model...")
ip = Input(shape=(X.shape[1],))

m = Dense(512, activation="relu")(ip)
m = Dropout(0.2)(m)
m = Dense(256, activation="relu")(m)
m = Dropout(0.2)(m)

op = Dense(y.shape[1], activation="softmax")(m)

model = Model(inputs=ip, outputs=op)

# Compile model
model.compile(
    optimizer='adam',
    loss="categorical_crossentropy", 
    metrics=['accuracy']
)

# Model summary
model.summary()

# Train model with validation
print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Evaluate the model
print("Evaluating model...")
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'training_history.png'))
print(f"Training history saved as 'training_history.png'")

# Generate confusion matrix
y_pred = np.argmax(model.predict(X_val), axis=1)
y_true = np.argmax(y_val, axis=1)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label, yticklabels=label)
plt.title('Confusion Matrix')
plt.ylabel('True Emotion')
plt.xlabel('Predicted Emotion')
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'confusion_matrix.png'))
print(f"Confusion matrix saved as 'confusion_matrix.png'")

# Save model and labels
model.save(os.path.join(os.getcwd(), "model.h5"))
np.save(os.path.join(os.getcwd(), "labels.npy"), np.array(label))

print("Training complete!")
print(f"Model saved as 'model.h5'")
print(f"Labels saved as 'labels.npy'")
