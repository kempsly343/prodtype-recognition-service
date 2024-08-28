from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json
import requests
from io import BytesIO
from PIL import Image
import cv2
import os

app = Flask(__name__)

# Configuration dictionary with image size and class names
CONFIGURATION = {
    "IM_SIZE": 256,
    "CLASS_NAMES": ["ACCESSORIES", "BRACELETS", "CHAIN", "CHARMS", "EARRINGS",
                    "ENGAGEMENT RINGS", "ENGAGEMENT SET", "FASHION RINGS", "NECKLACES", "WEDDING BANDS"],
}

# Global model and inference function
model = None
infer = None

def init_model():
    global model, infer
    model_path = 'models/lenet_model_save_tf/1'
    # Verify the files are present
    print("Files in model path:", os.listdir(model_path))
    # Load the TensorFlow SavedModel
    model = tf.saved_model.load(model_path)
    # Ensure that we can access the prediction function
    infer = model.signatures['serving_default']

def preprocess_image(image):
    # Resize and preprocess the image for the model
    image = cv2.resize(image, (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]))
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # Normalize to [0, 1]
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the input data
        data = request.get_json()
        
        if 'image_url' in data:
            # Load image from the URL
            response = requests.get(data['image_url'])
            image = np.array(Image.open(BytesIO(response.content)).convert('RGB'))
        elif 'image_data' in data:
            # Load image from the provided image data
            image = np.array(data['image_data'], dtype=np.uint8)
        else:
            return jsonify({"error": "No valid input image provided."}), 400
        
        # Preprocess the image
        image = preprocess_image(image)
        
        # Make predictions
        input_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        predictions = infer(input_tensor)['output_0']  # Adjust the key if necessary
        
        # Get the top 3 predicted classes and their probabilities
        predictions = predictions[0].numpy()
        top_3_indices = tf.argsort(predictions, direction='DESCENDING')[:3].numpy()
        top_3_probabilities = tf.gather(predictions, top_3_indices).numpy()
        top_3_classes = [CONFIGURATION['CLASS_NAMES'][index] for index in top_3_indices]
        
        # Prepare the results as a list of dictionaries
        top_3_predictions = [
            {"class_name": top_3_classes[i], "class_index": int(top_3_indices[i]), "probability": float(top_3_probabilities[i])}
            for i in range(3)
        ]
        
        # Return the JSON response
        return jsonify({"top_3_classes_predictions": top_3_predictions})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add the GET route to say "Hello"
@app.route('/', methods=['GET'])
def hello():
    return "Hello! Welcome to product type classification api awith Adaptives."

if __name__ == '__main__':
    init_model()  # Initialize the model before starting the server
    app.run()  # Defaults to host='127.0.0.1', port=5000
