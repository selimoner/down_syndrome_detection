import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import necessaryScripts

# Constants
IMAGE_SIZE = 224
BATCH_SIZE = 16
CHANNELS = 3
EPOCHS = 1

# Load data
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names

def get_dataset_partitions_tf(dataset, train_split=0.8, validation_split=0.1, test_split=0.1, shuffle=False, shuffle_size=10000):
    datasetLength = len(dataset)

    train_dataset_size = int(datasetLength * train_split)
    train_dataset = dataset.take(train_dataset_size)

    test_dataset = dataset.skip(train_dataset_size)

    validation_dataset_size = int(len(dataset) * validation_split)
    validation_dataset = test_dataset.take(validation_dataset_size)

    test_dataset = test_dataset.skip(validation_dataset_size)

    return train_dataset, validation_dataset, test_dataset

train_dataset, validation_dataset, test_dataset = get_dataset_partitions_tf(dataset)

train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomZoom(0.2, 0.2),
    layers.experimental.preprocessing.RandomContrast(0.2),
])

# Pre-trained model for feature extraction
#pretrained_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

# Define a new model with the pre-trained base
model = models.Sequential() #sıralı bir model oluşturuyoruz, kutunun içini dolduracağız
model.add(Conv2D(64, (3,3), activation="relu", padding="same", input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))) #conv2d katmanı ekliyoruz. içindeki 64 layer arasında geçiş yapılacak.
model.add(Conv2D(64, (3,3), activation="relu", padding="same"))
model.add(MaxPooling2D((2,2))) # gereksiz kısımları temizliyor
model.add(Dropout(0.25)) # 2.conv2d nin 64 katmanının 4te birini iptal et.
model.add(Conv2D(128, (3,3), activation="relu", padding="same"))
model.add(Conv2D(128, (3,3), activation="relu", padding="same"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(256, (3,3), activation="relu", padding="same"))
model.add(Conv2D(256, (3,3), activation="relu", padding="same"))
model.add(Conv2D(256, (3,3), activation="relu", padding="same"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(512, (3,3), activation="relu", padding="same"))
model.add(Conv2D(512, (3,3), activation="relu", padding="same"))
model.add(Conv2D(512, (3,3), activation="relu", padding="same"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(512, (3,3), activation="relu", padding="same"))
model.add(Conv2D(512, (3,3), activation="relu", padding="same"))
model.add(Conv2D(512, (3,3), activation="relu", padding="same"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Flatten()) # 3 channeldan 1 channel a indirgiyoruz.
model.add(Dense(4096,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4096,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1,activation="sigmoid")) # conv2d layerları tek bir dense layerına bağlanıyor ve sonucunda 1 veya 0 üretiliyor. (down send. veya değil)

# Build the model specifying the input shape
model.build((None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)) #modeli oluşturuyoruz.

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999) # adam en popüler optimizerlardan biri. içindeki değerler literatürde bu şekilde geçiyor.

# Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Compile the model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy']) # loss kısmı 1 veya 0 olacağından dolayı binary-crossentropy

# Train the model
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    verbose=1,
    validation_data=validation_dataset,
    callbacks=[early_stopping]
)

# Evaluate the model on the test dataset
scores = model.evaluate(test_dataset)
print("Test loss:", scores[0])
print("Test accuracy:", scores[1])

# Save the model
model_version = max([int(i) for i in os.listdir("../saved_models") + [0]]) + 1
model.save(f"../saved_models/preTrainedModel{necessaryScripts.getDateTime()}/no{model_version}")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

model_version = max([int(i) for i in os.listdir("../models") + [0]])+1
model.save(f"../models/pre_trained271023_2")

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
    plt.show()

