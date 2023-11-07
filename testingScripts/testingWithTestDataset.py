import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Constants
IMAGE_SIZE = 224
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 35

# Data Loading
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names

# Data Partitioning
def get_dataset_partitions_tf(dataset, train_split=0.8, validation_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    datasetLength = len(dataset)
    if shuffle:
        dataset = dataset.shuffle(shuffle_size, seed=12)

    train_dataset_size = int(datasetLength * train_split)
    train_dataset = dataset.take(train_dataset_size)

    test_dataset = dataset.skip(train_dataset_size)

    validation_dataset_size = int(len(dataset) * validation_split)
    validation_dataset = test_dataset.take(validation_dataset_size)

    test_dataset = test_dataset.skip(validation_dataset_size)

    return train_dataset, validation_dataset, test_dataset

train_dataset, validation_dataset, test_dataset = get_dataset_partitions_tf(dataset)

# Data Augmentation
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0 / 255)
])

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
    # Add more augmentation techniques here
])

# Load the pre-trained model
loaded_model = tf.keras.models.load_model(f"../models/pre_trained271023_2")

# Model Evaluation
scores = loaded_model.evaluate(test_dataset)
print("Test loss:", scores[0])
print("Test accuracy:", scores[1])

def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img.numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

image_path = "C:\\Users\\oners\\OneDrive\\Masaüstü\\590-down-syndrome-facial-features-s184-springer-high.jpg"
predicted_class , confidence = predict(loaded_model, image_path)
plt.imshow(image_path.numpy().astype('uint8'))
plt.title(f"Predicted: {predicted_class}\nConfidence: %{confidence}")
plt.axis("off")
plt.show()

"""
plt.figure(figsize=(15, 15))
for images, labels in test_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))

        predicted_class, confidence = predict(loaded_model, )
        actual_class = class_names[labels[i]]

        plt.title(f"Actual: {actual_class}\nPredicted: {predicted_class}\nConfidence: %{confidence}")

        plt.axis("off")
    plt.show()
"""