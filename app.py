from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)


model = tf.keras.models.load_model('plant_disease_model.h5')


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

class_labels = {v: k for k, v in class_indices.items()}

# Preprocessing function
def prepare_image(img):
    IMG_SIZE = 224  
    img = Image.open(io.BytesIO(img))
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return "Welcome to the Crop Disease Detection API!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        
        img = file.read()
        img_array = prepare_image(img)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        predicted_class = int(predicted_class)  

       
        disease_name = class_labels.get(predicted_class, 'Unknown')

       
        return jsonify({
            'prediction': disease_name,
            'confidence': str(np.max(predictions[0]))  
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
