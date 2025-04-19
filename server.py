import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model = load_model("model.h5")
class_labels = ["Residu", "Organik", "Anorganik"]

def preprocess_image(img):
    img = img.resize((224, 224))  
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        input_data = preprocess_image(img)
        prediction = model.predict(input_data)[0]

        predicted_class = class_labels[np.argmax(prediction)]
        probabilities = prediction.tolist()

        return jsonify({
            "prediction": predicted_class,
            "probabilities": probabilities
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))  
    app.run(host='0.0.0.0', port=port, debug=True)  
