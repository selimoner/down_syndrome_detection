import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import necessaryScripts
from PIL import Image

model = load_model("../saved_models/preTrainedModel256x256_161223______1500down1500normal")
class_names = ["Down Syndrome", "Normal"]

def preprocess_image(image_path):
    img = Image.open(image_path)
    face_list = necessaryScripts.mtcnnImage(img)

    resized_faces = []
    for face in face_list:
        # Resize the face to match the input size expected by the model
        resized_face = tf.image.resize(face, (256, 256))
        resized_faces.append(resized_face)

    return resized_faces

folder_path = "D:\\testing\\down"

down_counter = 0
normal_counter = 0

for images in os.listdir(folder_path):
    if images.endswith((".jpg", ".jpeg", ".png", ".PNG")):
        image_path = os.path.join(folder_path, images)
        img_array = preprocess_image(image_path)
        print(f"Faces found in image: {images} // Count: {len(img_array)}")

        if img_array is not None and len(img_array) > 0:
            counter = 1
            for image in img_array:
                predictions = model.predict(np.expand_dims(image, axis=0))
                predicted_class = np.argmax(predictions[0])
                if predicted_class == 0:
                    down_counter += 1
                elif predicted_class == 1:
                    normal_counter += 1
                confidence = round(100 * np.max(predictions[0]), 2)
                print(f"Image Name: {images} // Face no: {counter} // Predicted Class: {class_names[predicted_class]} // Confidence: {confidence}\n")
                counter += 1
        else:
            print(f"No faces found in image: {images}")

print(f"Down Syndrome Count: {down_counter} // Normal Count: {normal_counter}")
