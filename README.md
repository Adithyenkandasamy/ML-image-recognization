# ML-Image-Recognition

## 📌 Key Features
- ✅ Accepts image uploads for classification
- ✅ Uses a pre-trained deep learning model
- ✅ Provides confidence scores for each prediction
- ✅ Deployable as a web application using Flask

## 🛠 Technologies Used
- **TensorFlow & Keras** - Model training & prediction
- **Flask** - Web framework for deployment
- **Python** - Backend logic & scripting
- **NumPy & OpenCV** - Image preprocessing
- **HTML, CSS, JS** - Frontend (optional for UI)

## 📁 Project Structure
```
ML-Image-Recognition/
├── static/                 # Stores static assets (CSS, JS, images)
├── templates/              # HTML templates (if using a web interface)
├── train-images/           # Training dataset (if retraining)
├── validation-images/      # Validation dataset
├── app.py                  # Flask backend (for web app deployment)
├── main.py                 # Main script for training
├── test.py                 # Script for model inference
├── class_names.json        # Class labels for predictions
├── trained_model.h5        # Saved TensorFlow model
├── test_image1.jpeg        # Sample test image
├── test_image2.jpeg        # Sample test image
├── requirements.txt        # Required dependencies
├── README.md               # Documentation
```

## 🚀 Setup & Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Adithyenkandasamy/ML-image-recognition
cd ML-Image-Recognition
```

### 2️⃣ Create & Activate a Virtual Environment
```bash
python3 -m venv env
source env/bin/activate  # On Mac/Linux
env\Scripts\activate    # On Windows
```

### 3️⃣ Install Required Dependencies
```bash
pip install -r requirements.txt
```

## 🏋️ Training the Model
If you want to train the model from scratch, use `main.py`:
```bash
python main.py
```
This will:
- ✅ Load the dataset
- ✅ Train a convolutional neural network (CNN)
- ✅ Save the trained model as `trained_model.h5`

## 🖼 Running Image Classification
Use `test.py` to classify an image:
```bash
python test.py --image test_image1.jpeg
```
### Example Output:
```
'test_image2.jpeg' is classified as 'Laptop' with 99.88% confidence.
Backpack: 0.02%
Charger: 0.10%
Headphones: 0.00%
Laptop: 99.88%
Lock.and.key: 0.00%
Water bottle: 0.01%
```

## 🌐 Running the Flask Web Application
To run the web app, use:
```bash
python app.py
```
Then, open `http://127.0.0.1:5000/` in your browser.

## 🔗 GitHub Repository
[ML-Image-Recognition](https://github.com/Adithyenkandasamy/ML-image-recognition)


