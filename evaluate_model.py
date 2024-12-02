import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to Dataset and Model
test_dir = 'dataset/test'
model_path = 'saved_model/hand_classifier.h5'

# Image Dimensions
IMG_HEIGHT = 60
IMG_WIDTH = 80
BATCH_SIZE = 16

# Load Model
model = tf.keras.models.load_model(model_path)

# Data Generator for Testing
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_data = test_datagen.flow_from_directory(test_dir, target_size=(IMG_HEIGHT, IMG_WIDTH),
                                             batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)

# Evaluate Model
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy:.2f}")
