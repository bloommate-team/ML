import os
import numpy as np
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image


app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"] = set(['png','jpg','jpeg'])
app.config["UPLOAD_FOLDER"] = "static/uploads/"

def allowed_file(filename):
    return "." in filename and \
        filename.split(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]

model = load_model("modelCapstone.h5", compile=False)
with open("labels.txt", "r") as file:
    labels = file.read().splitlines()

@app.route("/")
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "Success",
        },
        "data": None
    }), 200

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            # save img
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            #pre processing
            img = Image.open(image_path).convert("RGB")
            img = img.resize((224,224))
            img_array = np.asarray(img)
            img_array = np.expand_dims(img_array, axis=0)
            normlized_image_array = (img_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)
            data[0] = normlized_image_array

            #predicting
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_names = labels[index]
            class_names = class_names[4:]
            confidence_score = prediction[0][index]

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success",
                }, 
                "data": {
                    "flower_name_prediction": class_names,
                    "confidence": float(confidence_score)
                }
            }), 200
        
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Error client side",
                }, 
                "data": None
            }), 400
    else:    
        return jsonify({
            "status": {
                "code": 405,
                "message": "Methods not allowed",
            },
            "data": None
        }), 405

if __name__ == "__main__":
    app.run()