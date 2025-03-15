import os
import json
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = load_model('trained_model.h5')

# Load class names correctly
class_file = 'class_names.json'
if os.path.exists(class_file):
    with open(class_file, 'r') as f:
        categories = json.load(f)
    if isinstance(categories, list):
        categories = {i: name for i, name in enumerate(categories)}
else:
    categories = {i: f"Class {i}" for i in range(model.output_shape[1])}

# Function to predict image class
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    predictions = model.predict(img_array)[0]
    max_prob = np.max(predictions)
    predicted_class_index = np.argmax(predictions)
    
    response = {
        "file": os.path.basename(img_path),
        "predicted_class": categories[predicted_class_index],
        "confidence": round(float(max_prob) * 100, 2),
        "class_probabilities": {categories[i]: round(float(prob) * 100, 2) for i, prob in enumerate(predictions)}
    }
    return response

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    result = predict_image(filepath)
    os.remove(filepath)  # Clean up after prediction
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
