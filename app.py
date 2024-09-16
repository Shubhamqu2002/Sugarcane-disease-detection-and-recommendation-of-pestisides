from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json

app = Flask(__name__)

# Load the trained model
model = load_model('models/Final_DenseNetSVM_Model.h5')

# Define disease prevention suggestions and pesticide recommendations
suggestions = {
    "Healthy": {
        "suggestion": "Your sugarcane is healthy. Keep up the good work!",
        "pesticides": "No pesticides needed."
    },
    "Red Dot": {
        "suggestion": "Improve soil drainage and avoid waterlogging. Apply appropriate fungicides.",
        "pesticides": "Suggested pesticides: Mancozeb, Carbendazim."
    },
    "Red Rust": {
        "suggestion": "Regularly inspect and prune affected leaves. Use rust-resistant varieties and apply fungicides.",
        "pesticides": "Suggested pesticides: Chlorothalonil, Propiconazole."
    }
}


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.densenet.preprocess_input(x)
    preds = model.predict(x)
    pred_class = np.argmax(preds, axis=1)
    # Adjust this according to your model's classes
    class_labels = ["Healthy", "Red Dot", "Red Rust"]
    return class_labels[pred_class[0]]


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        file_path = "uploads/" + file.filename
        file.save(file_path)
        prediction = model_predict(file_path, model)
        suggestion = suggestions[prediction]
        return jsonify({
            'prediction': prediction,
            'suggestion': suggestion['suggestion'],
            'pesticides': suggestion['pesticides']
        })


if __name__ == '__main__':
    app.run(debug=True)
