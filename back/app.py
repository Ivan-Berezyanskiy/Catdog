import os

import tensorflow as tf
from flask import Flask, request

from classifacator import classify

app = Flask(__name__)

STATIC_FOLDER = "static/"
UPLOAD_FOLDER = "static/upload/"

cnn_model = tf.keras.models.load_model(STATIC_FOLDER + "/models/" + "cat_dog.keras")


@app.get("/")
def home():
    return "Hello"


@app.post("/")
def upload_image():
    file = request.files["image"]
    upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)

    file.save(upload_image_path)

    label, prob = classify(cnn_model, upload_image_path)

    return {
        "label": label,
        "prob": prob
    }
