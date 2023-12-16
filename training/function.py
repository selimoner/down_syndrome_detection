import os
import cv2
from mtcnn.mtcnn import MTCNN
import time

def resize_images(input_folder, output_folder, target_size=(300, 300)):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    counter=0
    count=0

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            count+=1

    print(f"Total number of images : {count}\n")
    time.sleep(2)

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
        if counter==1000:
            break


if __name__ == "__main__":
    input_folder = "D:\\bitirmeProjesi\\data\\normal"
    output_folder = "D:\\bitirmeProjesi\\data\\normal2"

    target_size = (256, 256)

    resize_images(input_folder, output_folder, target_size)

