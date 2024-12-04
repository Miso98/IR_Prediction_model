import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Constants for image dimensions
IMG_HEIGHT = 60
IMG_WIDTH = 80

# Path to your saved model
model_path = '/home/mitchell/Documents/repos/IR_project/saved_model.h5'
model = load_model(model_path)

# Function to predict a single image
def predict_image(img_path):
    try:
        img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img)  # Convert image to array
        img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
        img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension

        prediction = model.predict(img_array)
        label = 'irregular' if prediction < 0.5 else 'regular'  # Assuming binary classification
        confidence = prediction[0][0]  # The output is a probability for the binary case
        print(f"Prediction: {label} (Confidence: {confidence:.2f})")

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# Directory containing your test images
test_dir = '/home/mitchell/Documents/repos/IR_project/dataset/test/regular/'

# Loop through files in the directory and predict
for filename in os.listdir(test_dir):
    file_path = os.path.join(test_dir, filename)
    
    # Only predict if the file is an image (optional, check file extension)
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        predict_image(file_path)
