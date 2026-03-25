import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from sklearn.model_selection import train_test_split
import joblib

print("Loading data...")
X = np.load('X_padded.npy')
y = np.load('y_labels.npy')

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

vocab_size = 10000
embedding_dim = 100  # Dimension of the dense embedding vector
max_length = 500

# 4. Build the 1D CNN Model
model = Sequential([
    # The Embedding layer learns a dense representation for each word
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),

    # Convolutional layer to extract n-gram features
    # filters=128 means it will learn 128 different phrase patterns
    # kernel_size=5 means it looks at 5 words at a time
    Conv1D(filters=128, kernel_size=5, activation='relu'),

    # Pool the most important features
    GlobalMaxPooling1D(),

    # Fully connected layers for classification
    Dense(64, activation='relu'),
    Dropout(0.5),  # Prevent overfitting

    # Output layer: 1 neuron with sigmoid for binary classification (Real vs Fake)
    Dense(1, activation='sigmoid')
])

# 5. Compile the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# 6. Train the Model
print("Starting training...")
history = model.fit(X_train, y_train,
                    epochs=5,
                    batch_size=64,
                    validation_data=(X_test, y_test))

# 7. Evaluate and Save
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

model.save('cnn_model.keras')
print("Model saved to cnn_model.keras")