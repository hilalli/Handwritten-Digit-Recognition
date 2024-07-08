import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

mnist = tf.keras.datasets.mnist  # handwritten character dataset 28x28

# After loading data, dividing it train and test datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Imported images are not in a shape that we want, they have RGB values so normalizing it makes it only black and white.
# Normalizing the dataset
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

IMG_SIZE = 28
x_trainr = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_testr = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Creating NN
model = Sequential()

# First Convolution layer: input shapes
model.add(Conv2D(64, (3, 3), input_shape=x_trainr.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolution layer
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third Convolution layer
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully connected layer 1
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

# Fully connected layer 2
model.add(Dense(32))
model.add(Activation("relu"))

# Last Fully Connected layer, output count = 10 since numbers are between 0-9
model.add(Dense(10))
model.add(Activation('softmax'))  # probability

# Compiling model
print("\nCompiling model...")
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
print("Model compiled successfully.")

# Training model
print("\nTraining model...")
model.fit(x_trainr, y_train, epochs=30, validation_split=0.3)
print("Model trained successfully.")

# Evaluating model
print("\nEvaluating model...")
test_loss, test_acc = model.evaluate(x_testr, y_test)
print("Loss:", test_loss)
print("Accuracy:", test_acc)
print("Model evaluated successfully.")

# Saving model for use it later
print("\nSaving model...")
file_name = 'handwritten_digit_recognition_model.h5'
model.save(file_name)
print(f"Model saved successfully as '{file_name}'.")

print("\nTraining section completed.")
