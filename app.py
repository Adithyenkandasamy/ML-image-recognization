from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('trained_model.h5')

# Define function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    filepath = os.path.join('static/uploads', file.filename)
    file.save(filepath)
    
    img_array = preprocess_image(filepath)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    
    label = "Water Bottle" if predicted_class[0] == 0 else "Laptop"
    
    return jsonify({'prediction': label, 'file_path': filepath})

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True,host="0.0.0.0")
