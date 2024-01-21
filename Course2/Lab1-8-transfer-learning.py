import os

import keras.activations
import numpy as np
from keras import layers, Model
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.src.preprocessing.image import ImageDataGenerator

from Course2.Utility import image_to_array, show_model_history

os.putenv('TF_ENABLE_ONEDNN_OPTS', '0')
pre_trained_model_file = './model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3),
    include_top=False,
    weights=None
)

pre_trained_model.load_weights(pre_trained_model_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

print(pre_trained_model.summary())
last_layer = pre_trained_model.get_layer('mixed7')
print(f'output shape {last_layer.output_shape}')
last_output = last_layer.output

#add Dense layer
x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation=keras.activations.relu)(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation=keras.activations.sigmoid)(x)

model = Model(pre_trained_model.inputs, x)
print(model.summary())

model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

# train generator
train_img_dir = './dataset/cats_and_dogs_filtered/train'
train_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
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
# train the model
history = model.fit(train_generator, steps_per_epoch=10, epochs=100, verbose=1, validation_data=val_generator,
                    validation_steps=8)


for img in os.listdir('./dataset/prediction'):
    print(f'for input {img}')
    img = image.load_img(os.path.join('./dataset/prediction', img), target_size=(150, 150))
    x = image.img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    classes = model.predict(images)
    print(classes)
    if classes[0] > 0.5:
        print('dog')
    else:
        print('cat')

show_model_history(history)

