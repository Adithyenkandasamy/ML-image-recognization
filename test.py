import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

test_folder = "/home/jinwoo/Desktop/ML-image-recognition/test_image2.jpeg"

# Load the trained model
model = load_model('trained_model.h5')

# Load class names correctly
class_file = 'class_names.json'
if os.path.exists(class_file):
    with open(class_file, 'r') as f:
        categories = json.load(f)

    # If it's a list, create a dictionary mapping indices to names
    if isinstance(categories, list):
        categories = {i: name for i, name in enumerate(categories)}
else:
    categories = {i: f"Class {i}" for i in range(model.output_shape[1])}  # Default class names

# Function to predict the image class
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"⚠️ File '{img_path}' not found.")
        return
    
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.  # Normalize
    predictions = model.predict(img_array)[0]  # Get the first (only) result

    # Get prediction probabilities
    max_prob = np.max(predictions)  # Get highest probability
    predicted_class_index = np.argmax(predictions)  # Get predicted class index

    # Check confidence threshold
    if max_prob >= 0.70:
        print(f"✅ '{os.path.basename(img_path)}' is classified as '{categories[predicted_class_index]}' with {max_prob:.2%} confidence.")
    else:
        print(f"❌ '{os.path.basename(img_path)}': I didn't know, sorry.")
    
    # Print probabilities for all classes
    for i, prob in enumerate(predictions):  # Extract the first prediction row
        print(f"{categories[i]}: {float(prob):.2%}")  # Convert prob to float

# Path to test image or folder
if os.path.isfile(test_folder):  # Single file case
    predict_image(test_folder)
elif os.path.isdir(test_folder):  # Directory case
    for img_name in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img_name)
        predict_image(img_path)
else:
    print("⚠️ Test folder or image not found. Please provide a valid image or directory.")
