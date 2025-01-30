import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("E:\\GitHub\\images_dataset\\saved_model\\image_classification_model.keras")

class_names = ['hoodie', 'pants', 'shirts', 'shoes', 'shorts']

def predict(image):
    img = image.resize((180, 180))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    return f"This image most likely belongs to {predicted_class} with a {confidence:.2f}% confidence."

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>Image Classification</h1>")
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(label="Upload Image", type="pil")
            submit_button = gr.Button("Submit")
        
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Prediction", interactive=False)
            
    submit_button.click(fn=predict, inputs=img_input, outputs=output_text)

demo.launch(share=True)