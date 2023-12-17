from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import necessaryScripts
import base64
import tensorflow as tf
import requests

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("../models/preTrainedModel256x256_161223______1500down1500normal")

class_names = ["Down Syndrome", "Normal"]

def read_file_as_image(data) -> np.ndarray:
    original_image = Image.open(BytesIO(data))
    face_list = necessaryScripts.mtcnnImage(original_image)
    return face_list, original_image

def get_image_url(image):
    # Convert the NumPy array to a PIL Image
    image_pil = Image.fromarray(image)

    # Convert the image to base64 and create a data URL
    image_bytes = BytesIO()
    image_pil.save(image_bytes, format="JPEG")
    image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{image_base64}"
    return image_url

@app.get("/ping")
async def ping():
    return "no info"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_list, original_image = read_file_as_image(await file.read())
    counter = 1
    predictions_list = []

    for image in image_list:
        if image is None:
            raise HTTPException(status_code=400, detail="No valid image data")

        img_batch = np.expand_dims(image, 0)

        if img_batch is None:
            raise HTTPException(status_code=400, detail="Failed to convert the image to a tensor")

        predictions = model.predict(img_batch)
        print(f"{model}")
        print("image no : ", counter, " ---- ", predictions)
        prediction = predictions[0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        image_url = get_image_url(image)

        predictions_list.append({
            "image no": counter,
            "class": predicted_class,
            "confidence": confidence,
            "image_url": image_url
        })
        counter += 1

    return predictions_list

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
