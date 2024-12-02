import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Paths to Dataset
train_dir = 'dataset/train'
val_dir = 'dataset/val'

# Image Dimensions
IMG_HEIGHT = 60
IMG_WIDTH = 80
BATCH_SIZE = 16

# Data Generators
train_datagen = ImageDataGenerator(rescale=1.0/255, rotation_range=30, width_shift_range=0.2,
                                   height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_data = train_datagen.flow_from_directory(train_dir, target_size=(IMG_HEIGHT, IMG_WIDTH),
                                               batch_size=BATCH_SIZE, class_mode='binary')
val_data = val_datagen.flow_from_directory(val_dir, target_size=(IMG_HEIGHT, IMG_WIDTH),
                                           batch_size=BATCH_SIZE, class_mode='binary')

# Build Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(train_data, validation_data=val_data, epochs=10)

# Save Model
model.save('saved_model/hand_classifier.h5')

# Plot Training Results
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
