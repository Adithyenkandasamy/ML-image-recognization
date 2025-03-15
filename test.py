from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json

# Load the trained model
model = load_model('trained_model.h5')

# Automatically extract class names from the model (if available)
if 'class_names.json' in model.optimizer.get_config():
    with open('class_names.json', 'r') as f:
        categories = json.load(f)  # Load class names
else:
    categories = [str(i) for i in range(model.output_shape[1])]  # Default: index numbers

# Function to predict the image class
def predict_image(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.  # Normalize

    # Get prediction probabilities
    predictions = model.predict(img_array)
    max_prob = np.max(predictions)  # Get highest probability
    predicted_class_index = np.argmax(predictions)  # Get predicted class index

    # Check confidence threshold
    if max_prob >= 0.40:
        print(f"Image '{img_path}' is classified as '{categories[predicted_class_index]}' with {max_prob:.2%} confidence.")
    else:
        print(f"Image '{img_path}': I didn't know, sorry.")

# Example: Pass the image path
image_path = '/home/jinwoo/Desktop/ML-image-recognition/test_image.jpeg'  # Change this to your image path
predict_image(image_path)
