import cv2
import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import os

# # Set seeds and environment
# random.seed(1)
# np.random.seed(1)
# tf.random.set_seed(1)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Load entire dataset
full_dataset = tf.keras.utils.image_dataset_from_directory(
    "Image",
    image_size=(224, 224),
    batch_size=32,
    label_mode="int",
    shuffle=True,
    seed=123
)

# Print number of classes
#print("Class names:", full_dataset.class_names)
#print("Number of classes:", len(full_dataset.class_names))

# Count total batches
total_batches = tf.data.experimental.cardinality(full_dataset).numpy()
train_size = int(0.7 * total_batches)
val_size = int(0.2 * total_batches)
test_size = total_batches - train_size - val_size

# Split dataset
train_ds = full_dataset.take(train_size)
val_ds = full_dataset.skip(train_size).take(val_size)
test_ds = full_dataset.skip(train_size + val_size)

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# Define VGG16-like model
model = keras.Sequential()

# Block 1
model.add(Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=(224, 224, 3)))
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Block 2
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Block 3
model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Block 4
model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Block 5
model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Top
model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dense(4096, activation="relu"))
model.add(Dense(3, activation="softmax"))  # 3 classes

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10  # You can adjust this
)

# Evaluate model
loss, accuracy = model.evaluate(test_ds)
print(f"\n✅ Test Accuracy: {accuracy:.2%}")
