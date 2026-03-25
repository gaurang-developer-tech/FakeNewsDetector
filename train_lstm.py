import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split

# 1. Load the preprocessed data
print("Loading data...")
X = np.load('X_padded.npy')
y = np.load('y_labels.npy')

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define Model Hyperparameters (matching keras_prep.py)
vocab_size = 10000
embedding_dim = 100
max_length = 500

# 4. Build the Bidirectional LSTM Model
model = Sequential([
    # Standard embedding layer
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),

    # Bidirectional LSTM: 64 units reading forward, 64 reading backward
    Bidirectional(LSTM(64)),

    # Fully connected layers for classification
    Dense(64, activation='relu'),
    Dropout(0.5),  # Crucial for LSTMs as they tend to overfit easily

    # Output layer for binary classification
    Dense(1, activation='sigmoid')
])

# 5. Compile the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# 6. Train the Model
print("Starting LSTM training...")
# Note: LSTMs take significantly longer to train per epoch than CNNs
history = model.fit(X_train, y_train,
                    epochs=5,
                    batch_size=64,
                    validation_data=(X_test, y_test))

# 7. Evaluate and Save
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

model.save('lstm_model.keras')
print("Model saved to lstm_model.keras")