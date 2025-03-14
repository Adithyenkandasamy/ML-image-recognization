from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('trained_model.h5')

# Load new image for prediction
img_path = '/home/jinwoo/Desktop/image_class/image.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.

# Make prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

if predicted_class[0] == 0:
    print("Water Bottle")
else:
    print("Lock and Key")
