import random

import keras
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from Utility import prepare_images_from_zip, image_to_array, random_display_images, visualize_layers

# prepare the train dataset
train_img_dir = './dataset/horse-or-human'
horse_files, human_files = prepare_images_from_zip('dataset/horse-or-human.zip', train_img_dir)

# random display images
random_display_images(horse_files, human_files)

# build the model
model = keras.models.Sequential([
    # the first convolution, 300px*300px with 3 bytes color
    keras.layers.Conv2D(16, (3, 3), activation=keras.activations.relu,
                        input_shape=(300, 300, 3)),
    keras.layers.MaxPooling2D(2, 2),
    # the second convolution
    keras.layers.Conv2D(32, (3, 3), activation=keras.activations.relu),
    keras.layers.MaxPooling2D(2, 2),
    # the third convolution
    keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu),
    keras.layers.MaxPooling2D(2, 2),
    # the forth convolution
    keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu),
    keras.layers.MaxPooling2D(2, 2),
    # the fifth convolution
    keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Flatten(),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)])

print(model.summary())

model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

# generator
train_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
    train_img_dir,
    target_size=(300, 300),  # all images will be resized to 300*300
    batch_size=128,
    class_mode='binary'  # human or horse, so binary
)
print(train_generator)

# train the model
history = model.fit(train_generator, steps_per_epoch=8, epochs=15, verbose=1)

# prepare predict data
predict_imgs = np.vstack((image_to_array('./dataset/predict_horse.jpg'), image_to_array('./dataset/predict_human.jpg')))
classes = model.predict(predict_imgs, batch_size=10)
print(f'Predict probability: {classes}')
print(f"Readable result: {['Human' if c[0] > 0.5 else 'Horse' for c in classes]}")

max_features_in_cnn_layers = 64

# visualize the CNN layers
visualize_layers(model, random.choice(horse_files + human_files), 64)
