import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('plant_disease_model.h5')

# Path to the test image
img_path = r'D:\New downloads\dataset\train\Apple___Cedar_apple_rust\image (997).JPG'  # Replace with the path to a sample image

# Preprocess the image
IMG_SIZE = 224  # Ensure this matches the size used in training
img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

# Map predicted_class to disease names (use the same order as your training labels)
class_indices = {
    "Apple__Apple_scab": 0,
    "Apple__Black_rot": 1,
    "Apple__Cedar_apple_rust": 2,
    "Apple__healthy": 3,
    "Background_without_leaves": 4,
    "Blueberry__healthy": 5,
    "Cherry__Powdery_mildew": 6,
    "Cherry__healthy": 7,
    "Corn__Common_rust": 8,
    "Corn__Northern_Leaf_Blight": 9,
    "Corn__healthy": 10,
    "Grape__Esca_(Black_Measles)": 11,
    "Grape__Black_rot": 12,
    "Grape__healthy": 13,
    "Peach__Bacterial_spot": 14,
    "Peach__healthy": 15,
    "Pepper,_bell__Bacterial_spot": 16,
    "Pepper,_bell__healthy": 17,
    "Potato__Early_blight": 18,
    "Potato__Late_blight": 19,
    "Potato__healthy": 20,
    "Raspberry__healthy": 21,
    "Soybean__healthy": 22,
    "Squash__Powdery_mildew": 23,
    "Strawberry__Leaf_scorch": 24,
    "Strawberry__healthy": 25,
    "Tomato__Bacterial_spot": 26,
    "Tomato__Early_blight": 27,
    "Tomato__Late_blight": 28,
    "Tomato__Leaf_Mold": 29,
    "Tomato__Septoria_leaf_spot": 30,
    "Tomato__Target_Spot": 31,
    "Tomato__Tomato_mosaic_virus": 32,
    "Tomato__healthy": 33
}
  # Example mapping
disease_name = list(class_indices.keys())[list(class_indices.values()).index(predicted_class)]

# Get prediction confidence
confidence = predictions[0][predicted_class]

print(f"The model predicts: {disease_name}")
print(f"Confidence: {confidence:.2%}")  # Formats confidence as percentage with 2 decimal places
