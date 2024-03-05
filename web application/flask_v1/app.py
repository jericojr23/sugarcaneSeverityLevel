from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load your GoogLeNet model from a .h5 file
model = load_model(
    r"D:\New_Fourth_Year\projectDesignGit\sugarcaneSeverityLevel\model\DenseNet\DenseNet-2024-01-07-14-37-25.h5"
)

# Class labels
class_labels = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        img_file = request.files["image"]
        img_path = "temp_image.jpg"
        img_file.save(img_path)

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)

        # Extract predicted class labels
        predicted_labels = [np.argmax(pred) for pred in predictions]
        predicted_class_names = [class_labels[label] for label in predicted_labels]

        return render_template("result.html", predictions=predicted_class_names)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
