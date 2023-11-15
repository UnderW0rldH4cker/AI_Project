# Importing the required libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generating some example data
# Features (input data)
X = np.array([[0, 0], [0, 0], [0, 0], [1, 1]])
# Labels (output data)
y = np.array([[0], [1], [1], [0]])

# Creating a simple neural network model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))  # Hidden layer with 4 neurons and ReLU activation
model.add(Dense(1, activation='sigmoid'))           # Output layer with 1 neuron and sigmoid activation

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
model.fit(X, y, epochs=1000, verbose=2)

# Making predictions
predictions = model.predict(X)
print("Predictions:")
print(predictions)
