from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import necessaryScripts
import requests
from fastapi import HTTPException
import matplotlib.pyplot as plt

app = FastAPI()

endpoint = "http://localhost:8503/v1/models/down_syndrome_detection_model:predict"

class_names = ["Down Syndrome", "Normal"]

def read_original_image(data) -> np.ndarray:
    image = np.array(necessaryScripts.loadImage(Image.open(BytesIO(data))))
    return image

def read_file_as_image(data) -> np.ndarray:

    image = np.array(necessaryScripts.loadImage(Image.open(BytesIO(data))))
    return image

@app.get("/ping")
async def ping():
    return "no info"

# First api code

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
        image_list = read_file_as_image(await file.read())
        counter = 1
        for image in image_list:
            if image is None:
                raise HTTPException(status_code=400, detail="No valid image data")

            img_batch = np.expand_dims(image, 0)

            if img_batch is None:
                raise HTTPException(status_code=400, detail="Failed to convert the image to a tensor")

            json_data = {
                "instances": img_batch.tolist()
            }
            response = requests.post(endpoint, json=json_data)
            print("image no : ",counter," ---- ",response.json())
            prediction = np.array(response.json()['predictions'][0])
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

            return {
                "image no : ": counter,
                "class": predicted_class,
                "confidence": float(confidence)
            }
            counter+=1


# api code to predict multiple faces.
"""
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    original_image = read_original_image(await file.read())
    image_list = read_file_as_image(await file.read())
    counter = 1
    predictions_list = []

    for image in image_list:
        if image is None:
            raise HTTPException(status_code=400, detail="No valid image data")

        img_batch = np.expand_dims(image, 0)

        if img_batch is None:
            raise HTTPException(status_code=400, detail="Failed to convert the image to a tensor")

        json_data = {
            "instances": img_batch.tolist()
        }
        response = requests.post(endpoint, json=json_data)
        print("image no : ", counter, " ---- ", response.json())
        prediction = np.array(response.json()['predictions'][0])
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        predictions_list.append({
            "image no": counter,
            "class": predicted_class,
            "confidence": confidence
        })
        counter += 1

    return predictions_list

"""
"""
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    original_image = read_original_image(await file.read())
    image_list = read_file_as_image(await file.read())
    counter = 1
    predictions_list = []

    for image in image_list:
        if image is None:
            raise HTTPException(status_code=400, detail="No valid image data")

        img_batch = np.expand_dims(image, 0)

        if img_batch is None:
            raise HTTPException(status_code=400, detail="Failed to convert the image to a tensor")

        json_data = {
            "instances": img_batch.tolist()
        }
        response = requests.post(endpoint, json=json_data)
        print("image no : ", counter, " ---- ", response.json())
        prediction = np.array(response.json()['predictions'][0])
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        predictions_list.append({
            "image no": counter,
            "class": predicted_class,
            "confidence": confidence
        })
        counter += 1

    no_of_images = len(image_list)
    no_of_images += 1  # adding original image
    plot_size = no_of_images / 2
    plt.figure(figsize=(plot_size * 5, plot_size * 5))  # Adjust the size multiplier based on your preference
    plt.subplot(plot_size, plot_size, 1)
    plt.imshow(original_image.astype('uint8'))  # Remove .numpy() as the original_image is already a NumPy array
    counter = 2
    for image in image_list:
        plt.subplot(plot_size, plot_size, counter)
        plt.imshow(image.astype('uint8'))  # Use image instead of image.numpy()
        image_number = predictions_list[counter - 2]['image no']
        predict_class = predictions_list[counter - 2]['class']
        confidence = predictions_list[counter - 2]['confidence']
        plt.title(f"Image No : {image_number}\n Class : {predict_class}\n Confidence : {confidence}")
        plt.axis("off")
        counter += 1

    plt.show()
    return predictions_list
"""

if __name__ == "__main__":
    uvicorn.run(app, host = 'localhost', port=8000)