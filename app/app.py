"""
app.py
Flask web app that:
  1. Accepts chest X-ray image upload
  2. Loads the trained EfficientNetB0 model
  3. Predicts NORMAL vs PNEUMONIA
  4. Generates a Grad-CAM heatmap overlay for explainability
"""

import os
import io
import base64
import uuid

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.efficientnet import preprocess_input
from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "pneumonia_model.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
IMG_SIZE = (224, 224)
LAST_CONV_LAYER_NAME = "top_conv"  # EfficientNetB0's last conv layer

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Lazy-load model so the app can still start even before training is done
model = None


def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Train it first using model/train.py or the Colab notebook, "
                "and place pneumonia_model.h5 inside the model/ folder."
            )
        model = tf.keras.models.load_model(MODEL_PATH)
    return model


def preprocess_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    array = img_to_array(img)
    array = preprocess_input(array)
    return np.expand_dims(array, axis=0)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(img_path, heatmap, alpha=0.4):
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    img_arr = np.array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap_img = Image.fromarray(np.uint8(jet_heatmap * 255)).resize(IMG_SIZE)
    jet_heatmap_arr = np.array(jet_heatmap_img)

    overlay = jet_heatmap_arr * alpha + img_arr * (1 - alpha)
    overlay = np.uint8(overlay)
    return Image.fromarray(overlay)


def image_to_base64(pil_img):
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        clf = get_model()
        img_array = preprocess_image(filepath)
        prediction = clf.predict(img_array)[0][0]

        label = "PNEUMONIA" if prediction >= 0.5 else "NORMAL"
        confidence = float(prediction) if label == "PNEUMONIA" else float(1 - prediction)

        heatmap = make_gradcam_heatmap(img_array, clf, LAST_CONV_LAYER_NAME)
        overlay_img = overlay_gradcam(filepath, heatmap)
        overlay_b64 = image_to_base64(overlay_img)

        return jsonify(
            {
                "label": label,
                "confidence": round(confidence * 100, 2),
                "gradcam_image": f"data:image/png;base64,{overlay_b64}",
            }
        )
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
