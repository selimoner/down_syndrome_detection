import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np
import os
import necessaryScripts

# Constants
IMAGE_SIZE = 224
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 40

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
    layers.experimental.preprocessing.RandomZoom(0.1),
    layers.experimental.preprocessing.RandomContrast(0.2),
    # You can add more augmentation techniques here
])

# Model Creation
input_shape = (BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,CHANNELS)
n_classes = 2
model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

model.build(input_shape=input_shape)
model.summary()

# Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=7,  # Number of epochs with no improvement after which training will stop
    restore_best_weights=True
)

# Training
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    verbose=1,
    validation_data=validation_dataset,
    callbacks=[early_stopping]  # Add early stopping callback
)

# Model Evaluation
scores = model.evaluate(test_dataset)
print("Test loss:", scores[0])
print("Test accuracy:", scores[1])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

model_version = max([int(i) for i in os.listdir("../models") + [0]])+1
model.save(f"../models/customModel{necessaryScripts.getDateTime()}/no{model_version}")

"""
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(len(acc), val_acc, label='Training Accuracy'))
plt.plot(range(len(acc), val_loss, label='Validation Loss'))
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy Data')

plt.subplot(1, 2, 2)
plt.plot(range(len(acc), loss, label='Training Loss'))
plt.plot(range(len(acc), val_loss, label='Validation Loss'))
plt.legend(loc='upper right')
plt.title('Training and Validation Loss Data')



# Tahmin etme
for images_batch, labels_batch in test_dataset.take(1):
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()

    print("First image to predict.")
    plt.imshow(first_image)
    print("First image's actual label : ",class_names[first_label])

    batch_prediction=model.predict(images_batch)
    print("Predicted label : ",class_names[np.argmax(batch_prediction[0])])

def predict(model,img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])),2)
    return predicted_class, confidence

plt.figure(figsize=(15,15))
for images, labels in test_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))

        predicted_class , confidence = predict(model, images[i])
        actual_class = class_names[labels[i]]

        plt.title(f"Actual : {actual_class}\nPredicted : {predicted_class}\nConfidence : %{confidence}")

        plt.axis("off")
        plt.show()"""