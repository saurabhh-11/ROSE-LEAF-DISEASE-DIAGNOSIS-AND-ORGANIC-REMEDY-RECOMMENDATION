import datetime
import os
import json
import numpy as np
from flask import Flask, logging, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from keras.models import load_model # type: ignore
import logging

app = Flask(__name__)


# Load your model
model = load_model('D:/project_1/IMPLEMENTATION/incpv3model.keras', compile=False)
print('Model loaded. Check http://127.0.0.1:5000/')

# Class labels
labels = {
    0: 'Black Spot',
    1: 'Downy Mildew',
    2: 'Fresh Leaf',
    3: 'Powdery Mildew',
    4: 'Rose Mosaic',
    5: 'Rose Rust',
    6: 'Rose Slug'
}

# Recommendations for each class
recommendations = {
    'Black Spot': {
        'Fertilizer': 'Compost or well-aged manure to strengthen plant immunity.',
        'Treatment': 'Baking soda spray (1 tbsp baking soda + 1 gallon of water) or neem oil.'
    },
    'Downy Mildew': {
        'Fertilizer': 'Rock phosphate or bone meal to improve root strength.',
        'Treatment': 'Copper-based fungicide or hydrogen peroxide spray (1 tbsp per quart of water).'
    },
    'Fresh Leaf': {
        'Fertilizer': 'Balanced organic fertilizer such as fish emulsion or compost tea to promote healthy growth.',
        'Treatment': 'No treatment needed. Ensure regular watering and sunlight.'
    },
    'Powdery Mildew': {
        'Fertilizer': 'Fish emulsion or compost tea to boost overall plant health.',
        'Treatment': 'Neem oil or a milk and water mixture (1:9 ratio).'
    },
    'Rose Mosaic': {
        'Fertilizer': 'Apply organic mulch (straw, leaves) and compost tea to support plant health.',
        'Treatment': 'Unfortunately, there is no cure for rose mosaic virus. Remove and destroy infected plants to prevent the spread.'
    },
    'Rose Rust': {
        'Fertilizer': 'Seaweed extract or alfalfa meal for better resistance.',
        'Treatment': 'Sulfur-based organic fungicide.'
    },
    'Rose Slug': {
        'Fertilizer': 'Fish emulsion to strengthen leaves and boost recovery.',
        'Treatment': 'Spray leaves with neem oil or insecticidal soap. Hand-picking or hosing them off with water also works.'
    }
}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create a temporary directory if it doesn't exist
TEMP_DIR = 'uploads'
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Path to the JSON file
JSON_FILE_PATH = 'predictions.json'

def load_predictions():
    if os.path.exists(JSON_FILE_PATH):
        with open(JSON_FILE_PATH, 'r') as f:
            return json.load(f)
    else:
        return []

def save_predictions(predictions):
    with open(JSON_FILE_PATH, 'w') as f:
        json.dump(predictions, f)

@app.route('/')
def index():
    return render_template('idx.html')

logging.basicConfig(level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        logging.error('No file part in the request')
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']

    if file.filename == '':
        logging.error('No file selected')
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(TEMP_DIR, filename)

        if os.path.exists(file_path):
            os.remove(file_path)
        
        try:
            file.save(file_path)
            logging.info(f"File {filename} saved for prediction.")
            
            # Image preprocessing
            img = load_img(file_path, target_size=(224, 224))
            logging.info(f"Image {filename} loaded and resized to (224, 224).")
            
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Add this line to normalize
            logging.info(f"Image array shape: {img_array.shape}")


            # Prediction
            prediction = model.predict(img_array)
            logging.info(f"Prediction made: {prediction}")
            
            predicted_class = np.argmax(prediction, axis=1)[0]
            logging.info(f"Predicted class: {predicted_class}")

            # Fetch corresponding disease, fertilizer, and treatment
            disease_name = labels[predicted_class]
            fertilizer = recommendations[disease_name]['Fertilizer']
            treatment = recommendations[disease_name]['Treatment']

            # Save prediction to JSON file
            predictions = load_predictions()
            predictions.append({
                'filename': filename,
                'disease_name': disease_name,
                'fertilizer': fertilizer,
                'treatment': treatment,
                'prediction_time': str(datetime.datetime.now())
            })
            save_predictions(predictions)
            logging.info(f"Prediction saved: {disease_name}, {fertilizer}, {treatment}")

            # Clean up: Delete the uploaded file after prediction
            os.remove(file_path)
            logging.info(f"File {filename} deleted after prediction.")

            return jsonify({
                'filename': filename,
                'disease_name': disease_name,
                'fertilizer': fertilizer,
                'treatment': treatment
            })

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return jsonify({'error': 'Prediction failed'}), 500

    else:
        logging.error('Invalid file type or empty file')
        return jsonify({'error': 'Invalid file type or empty file'}), 400

@app.route('/history', methods=['GET'])
def get_history():
    try:
        predictions = load_predictions()
        predictions.reverse()
        return jsonify(predictions)

    except Exception as e:
        print(f"Error occurred while fetching history: {e}")
        return jsonify({'error': 'Failed to fetch history'}), 500
    
@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        # Logic to clear the history (e.g., clearing the JSON file or database)
        with open('predictions.json', 'w') as file:
            json.dump([], file)  # Clear history by writing an empty list

        return jsonify({"success" : True}), 200
    except Exception as e:
        print(f"Error clearing history: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
