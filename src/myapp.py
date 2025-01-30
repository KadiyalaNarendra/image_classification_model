from flask import Flask, request, render_template
import os
import uuid
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore

app = Flask(__name__)

model = load_model("C:\\Users\\2276605\\OneDrive - Cognizant\\Documents\\Project SOLO\\Image Classification\\image_classification_model-main\\saved_model\\image_classification_model_new.keras")

class_names = ['hoodie', 'pants', 'shirts', 'shoes', 'shorts']

data_dir = "C:\\Users\\2276605\\OneDrive - Cognizant\\Documents\\Project SOLO\\Image Classification\\image_classification_model-main\\user_feedback_data"
os.makedirs(data_dir, exist_ok=True)
for class_name in class_names:
    os.makedirs(os.path.join(data_dir, class_name), exist_ok=True)

def predict_class(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize((180, 180))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis = 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    print(4)
    return predicted_class,confidence

def save_corrected_image(image_path, correct_label):
    label_dir = os.path.join(data_dir, correct_label)
    os.makedirs(label_dir, exist_ok=True)
    unique_filename = f"{uuid.uuid4()}.jpg"
    corrected_image_path = os.path.join(label_dir, unique_filename)
    Image.open(image_path).save(corrected_image_path)
    return corrected_image_path

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        print(1)
        if "file" in request.files:
            print(2)
            file = request.files["file"]
            if file and file.filename:
                print(3)
                upload_folder = "uploads"
                os.makedirs(upload_folder, exist_ok=True)
                image_path = os.path.join(upload_folder, file.filename)
                file.save(image_path)

                predicted_class, confidence = predict_class(image_path)
                print(predicted_class, confidence)
                
                return render_template("index.html", 
                                       step="result", 
                                       image_path=image_path, 
                                       predicted_class=predicted_class, 
                                       confidence=confidence)

        elif "feedback" in request.form:
            print(10)
            image_path = request.form["image_path"]
            predicted_class = request.form["predicted_class"]
            feedback = request.form["feedback"]

            if feedback == "Correct":
                message = f"The prediction {predicted_class} was confirmed as correct!"
            elif feedback == "Incorrect":
                correct_label = request.form["correct_label"]
                save_corrected_image(image_path, correct_label)
                message = f"The image has been saved with the correct label {correct_label}."
            else:
                message = "No feedback received."

            return render_template("index.html", step="feedback", message=message)

    return render_template("index.html", step="upload")

if __name__ == "__main__":
    app.run(debug=True)