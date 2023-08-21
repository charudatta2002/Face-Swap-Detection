import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Load your preprocessed data files
X_manipulated = np.load("Project\X_manipulated.npy")
X_original = np.load("Project\X_original.npy")

# Combine the X data arrays
X_normalized = np.concatenate((X_manipulated, X_original), axis=0)

# Load your label data files
y_manipulated = np.load("Project\y_manipulated.npy")
y_original = np.load("Project\y_original.npy")

# Combine the label arrays
y = np.concatenate((y_manipulated, y_original), axis=0)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Reshape the data for LSTM input
num_frames = 15  # Assuming each video is 15 frames long
num_features = 136  # Replace with the actual number of features
X_train_reshaped = X_train.reshape(X_train.shape[0], num_frames, num_features)
X_test_reshaped = X_test.reshape(X_test.shape[0], num_frames, num_features)

# Define the LSTM model
model = Sequential([
    LSTM(64, input_shape=(num_frames, num_features)),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_reshaped, y_train, batch_size=32, epochs=10, validation_data=(X_test_reshaped, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_reshaped, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


