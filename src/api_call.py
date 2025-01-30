from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore

app = Flask(__name__)

model = load_model("C:\\Users\\2276605\\OneDrive - Cognizant\\Documents\\Project SOLO\\Image Classification\\image_classification_model-main\\saved_model\\image_classification_model_new.keras")

class_names = ['hoodie', 'pants', 'shirts', 'shoes', 'shorts']

@app.route('/')
def index():
    return render_template("welcome.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['image']
    image = Image.open(file.stream)

    image = image.convert('RGB')
    image = image.resize((180, 180))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis = 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]

    return jsonify({
        'msg': 'success',
        'class': predicted_class
    })


if __name__ == "__main__":
    app.run(debug=True)