import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np

# Load Model
model_path = 'saved_model/hand_classifier.h5'
model = tf.keras.models.load_model(('/home/mitchell/Documents/repos/IR_project/saved_model.h5')
)

img_path = '/home/mitchell/Documents/repos/IR_project/dataset/test/regular/image1.png'

# Image Dimensions
IMG_HEIGHT = 60
IMG_WIDTH = 80

def predict_image(image_path):
    # Load and Preprocess Image
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make Prediction
    prediction = model.predict(img_array)
    class_name = 'irregular' if prediction[0] > 0.5 else 'regular'
    print(f"Prediction: {class_name} (Confidence: {prediction[0][0]:.2f})")

# Test Prediction
predict_image(img_path)

