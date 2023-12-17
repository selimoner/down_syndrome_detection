import tensorflow as tf
from tensorflow.keras import layers, models
"""Alexnet bir CNN modelidir. 5 adet konvolüsyon katmanı 3 adet tamamen bağlı katmanlardan oluşmaktadır."""

IMAGE_SIZE = 256  # AlexNet için önerilen giriş boyutu
BATCH_SIZE = 16
CHANNELS = 3
EPOCHS = 50

# Veri setimizi yüklüyoruz
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names

# Bütün dataseti %80 eğitim, %10 geçerleme ve %10 test verisi olacak şekilde bölüyoruz.
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

# Data Augmentation uyguluyoruz. Eğitim esnasında verileri döndürerek, yakınlaştırıp uzaklaştırarak ve kontrast değerleriyle oynayarak
# veri çeşitliliği sağlıyoruz.
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomZoom(0.2, 0.2),
    layers.experimental.preprocessing.RandomContrast(0.2),
])

"""
These layers consist of a series of convolutional and max-pooling layers. 
Each convolutional layer is responsible for learning hierarchical features from the input images. 
Max-pooling layers downsample the spatial dimensions of the feature maps, reducing the computational complexity. 
The activation function used is ReLU (Rectified Linear Unit), which introduces non-linearity.
"""
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

"""
These layers are fully connected (dense) layers. The flatten layer converts the 3D feature maps to a 1D vector. 
The two subsequent dense layers with ReLU activation introduce non-linearity, and dropout layers help prevent overfitting. 
The final dense layer with a sigmoid activation function outputs a single value between 0 and 1, 
representing the probability of the input image belonging to the positive class (binary classification).
"""

# Fully Connected Layer 1
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))

# Fully Connected Layer 2
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))

# Fully Connected Layer 3
model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification, so 1 output neuron with sigmoid activation

"""
The model is compiled using the Adam optimizer with a specified learning rate and momentum parameters. 
Binary crossentropy is chosen as the loss function for binary classification, 
and accuracy is used as the evaluation metric during training.
"""

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    verbose=1,
    validation_data=validation_dataset,
    callbacks=[early_stopping]
)

scores = model.evaluate(test_dataset)
print("Test loss:", scores[0])
print("Test accuracy:", scores[1])

model.save(f"../models/AlexNetStyleModel1")

for images_batch, labels_batch in test_dataset.take(1):
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()

    print("First image to predict.")
    plt.imshow(first_image)
    print("First image's actual label : ", class_names[first_label])

    batch_prediction = model.predict(images_batch)
    print("Predicted label : ", class_names[int(round(batch_prediction[0][0]))])
