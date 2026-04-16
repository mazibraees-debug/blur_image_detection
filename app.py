from flask import Flask, render_template, request
import onnxruntime as ort
import cv2
import numpy as np
import os
import base64

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===== Load ONNX Model =====
# We use onnxruntime for inference as it is much lighter than torch
session = ort.InferenceSession("blur_model.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ===== Prediction Function =====
def predict_image(img):
    # Preprocess: same as training (Resize to 128x128, normalize to [0,1])
    img_resized = cv2.resize(img, (128, 128)) / 255.0
    
    # Change format from HWC to CHW (standard for PyTorch/ONNX models)
    img_onnx = np.transpose(img_resized, (2, 0, 1)).astype(np.float32)
    
    # Add batch dimension (1, 3, 128, 128)
    img_onnx = np.expand_dims(img_onnx, axis=0)

    # Run inference
    outputs = session.run([output_name], {input_name: img_onnx})
    
    # Get prediction (argmax over class probabilities)
    pred = np.argmax(outputs[0], axis=1)[0]

    return "Blur" if pred == 0 else "Sharp"

# ===== Routes =====
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    img_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            # Vercel relies on a read-only filesystem, so we keep the image in memory
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is not None:
                result = predict_image(img)
                
                # Convert image to Base64 for the front-end to render without a disk url
                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                img_path = f"data:image/jpeg;base64,{img_base64}"

    return render_template("index.html", result=result, img_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)