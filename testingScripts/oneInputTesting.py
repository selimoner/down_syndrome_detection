import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import necessaryScripts
import cv2

model = load_model("../models/pre_trained271023_2")
class_names = ["Down Syndrome", "Normal"]

def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is not None:
        # Resize the image to 224x224
        image = cv2.resize(image, (224, 224))
        # Convert the image to a NumPy array and expand dimensions
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)
        return img_array
    else:
        return None

#image_path = "D:\\Software Engineering\\Projeler\\PycharmProjects\\down_syndrome_detection\\training\\data\\down_syndrome\\down_1.jpg"
#image_path = "D:\\Software Engineering\\Projeler\\PycharmProjects\\down_syndrome_detection\\training\\data\\normal\\normal_4.jpg"
folder_path = "/training/data/down_syndrome"

down_counter = 0
normal_counter = 0
for images in os.listdir(folder_path):
    if images.endswith((".jpg", ".jpeg", ".png", ".PNG")):
        image_path = os.path.join(folder_path, images)
        img_array = preprocess_image(image_path)
        if img_array is not None:
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            if predicted_class == 0:
                down_counter = down_counter+1
            elif predicted_class == 1:
                normal_counter = normal_counter+1
            confidence = round(100 * np.max(predictions[0]), 2)
            print(f"Image Name : {images} *///* Predicted Class : {class_names[predicted_class]} *///* Confidence : {confidence}\n")
            #print("Predicted Class:", class_names[predicted_class])
            #print("Confidence:", confidence)
        else:
            print("Invalid or missing image.")
print(f"Down Syndrome Count : {down_counter} // Normal Count : {normal_counter}")

"""
img_array2 = preprocess_image(image_path)

if img_array2 is not None:
    predictions = model.predict(img_array2)
    predicted_class = np.argmax(predictions[0])
    confidence = round(100 * np.max(predictions[0]), 2)

    print("Predicted Class:", class_names[predicted_class])
    print("Confidence:", confidence)
else:
    print("Invalid or missing image.")
"""
