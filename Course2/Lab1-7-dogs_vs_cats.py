import os

import keras
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from Utility import show_model_history, prepare_images_from_zip, random_display_images, image_to_array, visualize_layers

train_cat_files, train_dog_files, val_cat_files, val_dog_files = prepare_images_from_zip(
    './dataset/cats_and_dogs_filtered.zip', './dataset')
random_display_images(train_dog_files, train_cat_files)
random_display_images(val_dog_files, val_cat_files)

# train generator
train_img_dir = './dataset/cats_and_dogs_filtered/train'
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    train_img_dir,
    target_size=(150, 150),  # all images will be resized to 150*150
    batch_size=20,
    class_mode='binary'  # dog or cat, so binary
)
# validation generator
val_img_dir = './dataset/cats_and_dogs_filtered/validation'
val_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
    val_img_dir,
    target_size=(150, 150),  # all images will be resized to 150*150
    batch_size=20,
    class_mode='binary'  # dog or cat, so binary
)

# build the model
model = keras.models.Sequential([
    # the first convolution, 150px*150px with 3 bytes color
    keras.layers.Conv2D(16, (3, 3), activation=keras.activations.relu,
                        input_shape=(150, 150, 3)),
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

# train the model
history = model.fit(train_generator, steps_per_epoch=100, epochs=15, verbose=1, validation_data=val_generator,
                    validation_steps=8)

predit_imgs = os.listdir('./dataset/prediction')
print(f'Predit images: {predit_imgs}')
classes = model.predict(np.array([image_to_array(os.path.join('./dataset/prediction', img)) for img in predit_imgs]))
print(f'Predict probability: {classes}')
print(f"Readable result: {['Cat' if c[0] > 0.5 else 'Dog' for c in classes]}")
show_model_history(history)

visualize_layers(model, './dataset/prediction/cat-2.png', 64)
