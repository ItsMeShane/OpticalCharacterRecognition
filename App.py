from flask import Flask, request, jsonify, render_template
from NeuralNetwork import NN
from PIL import Image
import io
import base64
import numpy as np

app = Flask(__name__)

model = NN(28*28, 300, 10)
model.load_model('trained_model.pkl')

def load_and_preprocess_image(img):
    img_array = np.array(img) / 255.0 # convert the image to a numpy array and normalize it
    img_array = 1 - img_array # Invert the image 
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    img_data = request.json['image'].split(',')[1]  # Extract base64 data
    img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert('L')  # Decode and convert to grayscale

    # Preprocess image and make prediction
    img_array = load_and_preprocess_image(img)
    prediction_probabilities  = model.predict(img_array.reshape(1, -1))

    return jsonify({
        'probabilities': prediction_probabilities.tolist(),  # Send all probabilities
    })


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
