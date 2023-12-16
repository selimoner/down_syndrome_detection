import os
import cv2
import numpy as np
from PIL import Image
import datetime
from io import BytesIO
from mtcnn.mtcnn import MTCNN

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
"""
def loadImage(data):
    image = cv2.cvtColor(np.array(data), cv2.COLOR_RGB2BGR)
    height, width, channels = image.shape

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if(len(faces)==0):
        print("No faces found on the image.")
        return cv2.resize(image,(224,224))
    else:
        if height > 224 and width > 224:
            for (x, y, w, h) in faces:
                y -= int(0.2 * h)
                h += int(0.3 * h)
                cropped_face = image[y:y + h, x:x + w]
                resized_cropped_face = cv2.resize(cropped_face, (224, 224)
                return resized_cropped_face
    
    if height > 224 and width > 224:
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
"""


def loadImage(data):
    image = cv2.cvtColor(np.array(data), cv2.COLOR_RGB2BGR)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    detected_faces = []

    if len(faces) == 0:
        print("No faces found on the image.")
        resized_image = cv2.resize(image, (224, 224))
        detected_faces.append(resized_image)
        return detected_faces
    else:
        for (x, y, w, h) in faces:
            y -= int(0.2 * h)
            h += int(0.3 * h)
            y = max(0, y)
            h = max(1, h)
            cropped_face = image[y:y + h, x:x + w]
            resized_cropped_face = cv2.resize(cropped_face, (224, 224))
            detected_faces.append(resized_cropped_face)

        return detected_faces

"""def mtcnnImage(data):
    #image = cv2.cvtColor(np.array(data), cv2.COLOR_RGB2BGR)
    # Convert the input data to a NumPy array
    image = np.array(data)

    # Check if the image has 4 channels and convert to RGB if needed
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Create an MTCNN face detector
    detector = MTCNN()

    # Detect faces
    faces = detector.detect_faces(image)

    detected_faces = []
    detected_faces2=[]

    for face in faces:
        x, y, w, h = face['box']
        confidence = face['confidence']

        if confidence >= 0.5:  # Adjust confidence threshold as needed
            # Add a margin to the bounding box to get a larger face area
            margin = 0.3
            x -= int(w * margin)
            y -= int(h * margin)
            w += int(2 * w * margin)
            h += int(2 * h * margin)

            # Ensure the modified bounding box is within the image boundaries
            x, y = max(0, x), max(0, y)
            w, h = min(image.shape[1] - 1, w), min(image.shape[0] - 1, h)
            # Extract and resize the modified face area
            cropped_face = image[y:y + h, x:x + w]
            resized_cropped_face = cv2.resize(cropped_face, (256, 256))
            detected_faces.append(resized_cropped_face)




    if len(detected_faces) == 0:
        detected_faces = loadImage(data)  # Assuming you have a function to load the image

    return detected_faces
"""


def mtcnnImage(data):
    # Convert the input data to a NumPy array
    image = np.array(data)

    # Check if the image has 4 channels and convert to RGB if needed
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Create an MTCNN face detector
    detector = MTCNN()

    # First pass: Detect faces
    faces = detector.detect_faces(image)

    detected_faces = []

    for face in faces:
        x, y, w, h = face['box']
        confidence = face['confidence']

        if confidence >= 0.5:  # Adjust confidence threshold as needed
            # Extract and resize the face area
            cropped_face = image[y:y + h, x:x + w]
            resized_cropped_face = cv2.resize(cropped_face, (256, 256))
            detected_faces.append(resized_cropped_face)

    # Second pass: Detect faces on the already detected faces
    second_pass_detected_faces = []

    for face in detected_faces:
        # Convert the face to RGB format if needed
        if face.shape[-1] == 4:
            face = cv2.cvtColor(face, cv2.COLOR_RGBA2RGB)

        # Detect faces on the already detected face
        second_pass_faces = detector.detect_faces(face)

        for second_pass_face in second_pass_faces:
            x, y, w, h = second_pass_face['box']
            confidence = second_pass_face['confidence']

            if confidence >= 0.5:  # Adjust confidence threshold as needed
                # Extract and resize the face area
                second_pass_cropped_face = face[y:y + h, x:x + w]
                second_pass_resized_cropped_face = cv2.resize(second_pass_cropped_face, (256, 256))
                second_pass_detected_faces.append(second_pass_resized_cropped_face)

    if len(second_pass_detected_faces) == 0:
        second_pass_detected_faces = loadImage(data)  # Assuming you have a function to load the image

    return second_pass_detected_faces
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