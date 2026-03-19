import os
import io
import uvicorn
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'deepfake_mobilenetv2_model.h5')
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read and open (Matches Streamlit)
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # 2. Resize → Array (raw float32 – no preprocess_input needed for this model)
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
    
    # 3. Predict
    prediction = model.predict(img_array)
    score = float(prediction[0][0])
    
    # Debug: Watch your terminal
    print(f"DEBUG - Raw Prediction Score: {score}")
    
    # ── Classification — matches the original Streamlit convention ────────────
    # Model trained with sigmoid output:  1.0 = Real face,  0.0 = Fake / AI-generated
    # score > 0.5  →  Real   (confidence = how high the score is, i.e. score × 100)
    # score ≤ 0.5  →  Fake   (confidence = how far below 0.5 it is, (1 − score) × 100)
    if score > 0.5:
        label      = "Real"
        confidence = round(score * 100, 1)           # e.g. 0.90 → 90 %
    else:
        label      = "Fake"
        confidence = round((1.0 - score) * 100, 1)  # e.g. 0.03 → 97 %

    print(f"DEBUG  score={score:.4f}  →  {label}  ({confidence}%)")

    return {
        "label":     label,
        "confidence": confidence,
        "raw_score":  round(score, 4)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
