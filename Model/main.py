from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import io
import pickle

# ---------- Load model and label encoder ----------
model = load_model("/Users/shashwatshrivastava/Downloads/cnn_model.h5")
with open("/Users/shashwatshrivastava/Downloads/label_encoder.pkl", "rb") as f:

    label_encoder = pickle.load(f)

IMG_SIZE = 160  # same as used during training

# ---------- Create FastAPI app ----------
app = FastAPI()

def preprocess_image(file) -> np.ndarray:
    img = Image.open(io.BytesIO(file)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 160, 160, 3)
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_array = preprocess_image(contents)
        preds = model.predict(img_array)
        class_index = np.argmax(preds, axis=1)[0]
        class_name = label_encoder.inverse_transform([class_index])[0]
        confidence = float(np.max(preds))

        return {
            "predicted_class": class_name,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

