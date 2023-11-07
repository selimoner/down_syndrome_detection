import os
import cv2
import numpy as np
from PIL import Image
import datetime
from io import BytesIO

folder_path = r"C:\Users\oners\OneDrive\Masaüstü\testing"

def rename_images(file_path):
    counter = 1
    for filename in os.listdir(folder_path):
        file_extension = filename[-4:]
        if file_extension.lower() in {'.jpg', '.png', '.jpeg'}:
            old_file_path = os.path.join(folder_path, filename)
            new_file_name = "normal_{0}{1}".format(counter, file_extension)
            new_file_path = os.path.join(folder_path, new_file_name)
            os.rename(old_file_path, new_file_path)
            counter += 1


def find_face_resize(folder_path):
    counter=1
    for image_name in os.listdir(folder_path):
        if image_name.endswith((".jpg", ".jpeg", ".png", ".PNG")):
            image_path = os.path.join(folder_path,image_name)
            image_path.decode('utf8')
            #image_path = f"{str(folder_path)}\\{str(image_name)}"
            image = cv2.imread(image_path)
            print(image_path)
            print(image)
            if image is not None:
                height, width, channels = image.shape

                if height > 224 and width > 224:
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    for (x, y, w, h) in faces:
                        # Adjust the cropping area to include the upper body
                        y -= int(0.2 * h)
                        h += int(0.3 * h)

                        # Crop the face and upper body
                        cropped_face = image[y:y + h, x:x + w]

                        # Resize the cropped image to 224x224
                        resized_cropped_face = cv2.resize(cropped_face, (224, 224))

                        # Save or display the resulting cropped and resized image
                        cv2.imwrite(f"cropped_face{counter}.jpg",
                                    resized_cropped_face)  # Save the cropped image to a file
                        cv2.imshow("Cropped Face", resized_cropped_face)  # Display the cropped image
                        cv2.waitKey(0)  # Wait for a key press to close the display window
                        cv2.destroyAllWindows()  # Close all OpenCV windows
            else:
                print("Can't find the image.")

def findInputFaceAndResize(image_path):
    if image_path.endswith((".jpg", ".jpeg", ".png", ".PNG")):
        image = cv2.imread(image_path)
        if image is not None:
            height, width, channels = image.shape
            if height > 224 and width > 224:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    y -= int(0.2 * h)
                    h += int(0.3 * h)
                    cropped_face = image[y:y + h, x:x + w]
                    resized_cropped_face = cv2.resize(cropped_face, (224, 224))
                    print(resized_cropped_face.shape)
                    return resized_cropped_face

def loadImage(data):
    image = cv2.cvtColor(np.array(data), cv2.COLOR_RGB2BGR)
    height, width, channels = image.shape
    if height > 224 and width > 224:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        print(f"Faces found : {len(faces)}")
        if len(faces) == 0:
            return cv2.resize(image, (224,224))
        for (x, y, w, h) in faces:
            y -= int(0.2 * h)
            h += int(0.3 * h)
            cropped_face = image[y:y + h, x:x + w]
            resized_cropped_face = cv2.resize(cropped_face, (224, 224))
            print(resized_cropped_face.shape)
            return resized_cropped_face
    return cv2.resize(image, (224,224))

def getDateTime():
    current_time = datetime.datetime.now()
    day = current_time.day
    month = current_time.month
    year = current_time.year
    hour = current_time.hour
    minute = current_time.minute
    model_name = f"{day}-{month}-{year}/{hour}-{minute}"
    return model_name

#rename_images(folder_path)
#find_face_resize(folder_path)