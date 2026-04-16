from flask import Flask, render_template, request
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import base64

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===== Model (same as training) =====
class BlurClassifier(nn.Module):
    def __init__(self):
        super(BlurClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ===== Load Model =====
device = torch.device("cpu")
model = BlurClassifier().to(device)
model.load_state_dict(torch.load("blur_model.pth", map_location=device, weights_only=True))
model.eval()

# ===== Prediction Function =====
def predict_image(img):
    img = cv2.resize(img, (128, 128)) / 255.0
    img = torch.tensor(img).permute(2,0,1).float().unsqueeze(0)

    device_target = model.conv[0].weight.device
    img = img.to(device_target)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()

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