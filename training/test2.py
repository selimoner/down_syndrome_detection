import tensorflow as tf
from tensorflow.keras import layers, models

# Constants
IMAGE_SIZE = 256  # AlexNet için önerilen giriş boyutu
BATCH_SIZE = 16
CHANNELS = 3
EPOCHS = 50

# Load data
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names

# Function to get dataset partitions
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

# AlexNet benzeri model
model = models.Sequential()
model.add(data_augmentation)

# Convolutional Layer 1
model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)))
model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

# Convolutional Layer 2
model.add(layers.Conv2D(256, (5, 5), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

# Convolutional Layer 3
model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))

# Convolutional Layer 4
model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))

# Convolutional Layer 5
model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

# Fully Connected Layer 1
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))

# Fully Connected Layer 2
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))

# Fully Connected Layer 3
model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification, so 1 output neuron with sigmoid activation

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

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

model.save(f"../models/AlexNetStyleModel1")

# Prediction example
for images_batch, labels_batch in test_dataset.take(1):
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()

    print("First image to predict.")
    plt.imshow(first_image)
    print("First image's actual label : ", class_names[first_label])

    batch_prediction = model.predict(images_batch)
    print("Predicted label : ", class_names[int(round(batch_prediction[0][0]))])
