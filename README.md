# Blur Image Detection

This is a Flask web application that determines whether an uploaded image is "Blur" or "Sharp" using a PyTorch classifier model.

## Features
- **Upload Image**: Drag or select an image to test the model.
- **Image Visualization**: Gives a user-friendly UI preview of the image right after uploading.
- **Serverless Ready Architecture**: Bypasses the local file system by keeping parsed images directly in memory and visualizing them with Base64 strings. This allows seamless transitions onto Vercel Serverless deployments.

## Setup Instructions

### 1. Requirements

Install all requirements in an activated `venv`:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Add Model File

You need a trained PyTorch model binary `blur_model.pth` located at the root of the project. This model binary defines the parameters corresponding to `BlurClassifier(nn.Module)`.

### 3. Start Local Environment

```bash
python app.py
```
Your server will deploy locally. Check `http://127.0.0.1:5000` to interact.

---

## Deploying to Vercel

Deployment configuration has been optimized securely in `vercel.json` and memory buffers logic inside `app.py`.

1. To begin, establish a Vercel Project if one does not exist or integrate straight from an environment like Git using the new `vercel.json`.
2. A `.gitignore` file exists. Commit to a remote branch and go to `vercel.com/new` linking the repository.

**Note about PyTorch Limitations via Vercel**:
Vercel's serverless function hard limit size is `250 MB` total (uncompressed). Since `torch` and `opencv-python` can be very large, deployment deployments can occasionally throw size limits errors if the required build output bypasses sizes limits. Should this happen:
- The backend API processing `torch` inference generally needs to be detached to a separate platform (like AWS, Render, or Hugging Face Spaces) leaving the UI/BFF hosted inside Vercel.
