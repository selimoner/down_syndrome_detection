import os
import cv2
import numpy as np
import datetime
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

def oneFaceDetection(data):
    image = np.array(data)

    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    detector = MTCNN()

    faces = detector.detect_faces(image)

    detected_faces = []

    for face in faces:
        x, y, w, h = face['box']
        confidence = face['confidence']

        if confidence >= 0.5:
            cropped_face = image[y:y + h, x:x + w]
            resized_cropped_face = cv2.resize(cropped_face, (256, 256))
            detected_faces.append(resized_cropped_face)

    if len(detected_faces) == 0:
        detected_faces = cv2.resize(image, (256, 256))

    return detected_faces

def mtcnnImage(data):
    image = np.array(data)

    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    detector = MTCNN()

    faces = detector.detect_faces(image)

    detected_faces = []

    for face in faces:
        x, y, w, h = face['box']
        confidence = face['confidence']

        if confidence >= 0.5:
            cropped_face = image[y:y + h, x:x + w]
            resized_cropped_face = cv2.resize(cropped_face, (256, 256))
            detected_faces.append(resized_cropped_face)

    second_pass_detected_faces = []

    for face in detected_faces:
        if face.shape[-1] == 4:
            face = cv2.cvtColor(face, cv2.COLOR_RGBA2RGB)

        second_pass_faces = detector.detect_faces(face)

        for second_pass_face in second_pass_faces:
            x, y, w, h = second_pass_face['box']
            confidence = second_pass_face['confidence']

            if confidence >= 0.5:
                second_pass_cropped_face = face[y:y + h, x:x + w]
                second_pass_resized_cropped_face = cv2.resize(second_pass_cropped_face, (256, 256))
                second_pass_detected_faces.append(second_pass_resized_cropped_face)

    if len(second_pass_detected_faces) == 0:
        second_pass_detected_faces = oneFaceDetection(data)

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

def resize_images(input_folder, output_folder, target_size=(300, 300)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    counter=0
    count=0

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            count+=1

    print(f"Total number of images : {count}\n")

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            img = cv2.imread(input_path)
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            detector = MTCNN()
            faces = detector.detect_faces(img)

            detected_faces = []

            for face in faces:
                x, y, w, h = face['box']
                confidence = face['confidence']

                if confidence >= 0.5:
                    cropped_face = img[y:y + h, x:x + w]
                    resized_cropped_face = cv2.resize(cropped_face, target_size[::-1])
                    detected_faces.append(resized_cropped_face)

            if len(detected_faces) == 1:
                resized_img = cv2.resize(detected_faces[0], target_size[::-1])
                print(f"\nChanged and Resized image : {filename}\n")
            else:
                resized_img = cv2.resize(img, target_size[::-1])
                print(f"\nOnly Resized image : {filename}")
            cv2.imwrite(output_path, resized_img)
        count-=1
        print(f"Images left : {count}\n")
        counter+=1
        print(f"Counter at : {counter}")
        if counter==1500:
            break

if __name__ == "__main__":
    input_folder = "D:\\bitirmeProjesi\\data\\normal"
    output_folder = "D:\\bitirmeProjesi\\data\\normal2"
    target_size = (256, 256)
    resize_images(input_folder, output_folder, target_size)