import os

import tensorflow as tf
from flask import Flask, request, render_template

from classificator import classify

STATIC_FOLDER = "web/static/"
TEMPLATES_FOLDER = "web/templates/"


app = Flask(__name__,
            template_folder="../" + TEMPLATES_FOLDER,
            static_url_path="",
            static_folder="../" + STATIC_FOLDER,
            )

UPLOAD_FOLDER = STATIC_FOLDER + "upload/"

cnn_model = tf.keras.models.load_model(
    STATIC_FOLDER + "/models/" + "cat_dog.keras"
)


@app.get("/")
def home():
    return render_template("home.html")


@app.post("/")
def upload_image():
    file = request.files["image"]
    upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)

    file.save(upload_image_path)

    label, prob = classify(cnn_model, upload_image_path)

    return render_template("classification-res.html", label=label, prob=prob)
