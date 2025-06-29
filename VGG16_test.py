import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# ──────────────────── DATA ────────────────────
data_dir   = "Dataset"
batch_size = 32
img_height = 224
img_width  = 224

full_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="int",
    shuffle=True,
    seed=123
)

total_batches = tf.data.experimental.cardinality(full_dataset).numpy()
train_size = int(0.7 * total_batches)
val_size   = int(0.2 * total_batches)
test_size  = total_batches - train_size - val_size

train_ds = full_dataset.take(train_size)
val_ds   = full_dataset.skip(train_size).take(val_size)
test_ds  = full_dataset.skip(train_size + val_size)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.prefetch(buffer_size=AUTOTUNE)

# ──────────────────── MODEL (VGG‑style) ────────────────────
model = Sequential([
    # Block 1
    Conv2D(64,  (3,3), padding="same", activation="relu", input_shape=(img_height, img_width, 3)),
    Conv2D(64,  (3,3), padding="same", activation="relu"),
    MaxPooling2D((2,2)),

    # Block 2
    Conv2D(128, (3,3), padding="same", activation="relu"),
    Conv2D(128, (3,3), padding="same", activation="relu"),
    MaxPooling2D((2,2)),

    # Block 3
    Conv2D(256, (3,3), padding="same", activation="relu"),
    Conv2D(256, (3,3), padding="same", activation="relu"),
    Conv2D(256, (3,3), padding="same", activation="relu"),
    MaxPooling2D((2,2)),

    # Block 4
    Conv2D(512, (3,3), padding="same", activation="relu"),
    Conv2D(512, (3,3), padding="same", activation="relu"),
    Conv2D(512, (3,3), padding="same", activation="relu"),
    MaxPooling2D((2,2)),

    # Block 5
    Conv2D(512, (3,3), padding="same", activation="relu"),
    Conv2D(512, (3,3), padding="same", activation="relu"),
    Conv2D(512, (3,3), padding="same", activation="relu"),
    MaxPooling2D((2,2)),

    # Top
    Flatten(),
    Dense(4096, activation="relu"),
    Dense(4096, activation="relu"),
    Dense(3,    activation="softmax")          
])

# ──────────────────── OPTIMIZER (SGD) ────────────────────
initial_lr = .000001                             
optimizer  = tf.keras.optimizers.SGD(            
    learning_rate=initial_lr,
    momentum=0.9,
    nesterov=False
)

model.compile(
    optimizer=optimizer,                         
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
model.summary()

# ──────────────────── TRAIN ────────────────────
epochs  = 100
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# ──────────────────── TEST ────────────────────
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n✅ Test Accuracy: {test_acc:.2%}")

# ──────────────────── PLOTS ────────────────────
acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']

import matplotlib.pyplot as plt
epochs_range = range(epochs)
plt.figure(figsize=(8,8))

plt.subplot(1,2,1)
plt.plot(epochs_range, acc,     label='Train Acc')
plt.plot(epochs_range, val_acc, label='Val Acc')
plt.legend(loc='lower right')
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss,     label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.legend(loc='upper right')
plt.title('Loss')
plt.show()
