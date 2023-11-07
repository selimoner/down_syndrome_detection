from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import necessaryScripts

app = FastAPI()

model = tf.keras.models.load_model("../saved_models/pre_trained271023_2")
class_names = ["Down Syndrome", "Normal"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(necessaryScripts.loadImage(Image.open(BytesIO(data))))
    return image

@app.get("/ping")
async def ping():
    return "no info"


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    if image is None:
        return {"error": "No valid image data"}

    img_batch = np.expand_dims(image, 0)

    if img_batch is None:
        return {"error": "Failed to convert the image to a tensor"}

    prediction = model.predict(img_batch)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host = 'localhost', port=8000)
