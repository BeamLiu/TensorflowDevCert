import zipfile
import os
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# prepare the dataset
images_dir = './dataset/horse-or-human'
if not os.path.exists(images_dir) or not any(os.listdir(images_dir)):
    print('extracting the images')
    zip_ref = zipfile.ZipFile('dataset/horse-or-human.zip')
    zip_ref.extractall(images_dir)
    zip_ref.close()

print('the images are prepared!')

train_horse_dir = os.path.join(f'{images_dir}/horses')
train_human_dir = os.path.join(f'{images_dir}/humans')

horse_files = [file for file in os.listdir(train_horse_dir) if os.path.isfile(os.path.join(train_horse_dir, file))]
human_files = [file for file in os.listdir(train_human_dir) if os.path.isfile(os.path.join(train_human_dir, file))]
print(f'total horse images {len(horse_files)}')
print(f'total human images {len(human_files)}')

# list first 8 images for each
f, axs = plt.subplots(2, 8, figsize=(32, 12))
for i in range(0, 8):
    axs[0, i].axis('off')
    axs[1, i].axis('off')
    axs[0, i].imshow(matplotlib.image.imread(os.path.join(train_horse_dir, horse_files[i])))
    axs[1, i].imshow(matplotlib.image.imread(os.path.join(train_human_dir, human_files[i])))

plt.show()

# build the model
model = keras.models.Sequential([
    # the first convolution, 300px*300px with 3 bytes color
    keras.layers.Conv2D(16, (3, 3), activation=keras.activations.relu,
                        input_shape=(300, 300, 3)),
    keras.layers.MaxPooling2D(2, 2),
    # the second convolution
    keras.layers.Conv2D(32, (3, 3), activation=keras.activations.relu),
    keras.layers.MaxPooling2D(2, 2),
    # the second convolution
    keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu),
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
    images_dir,
    target_size=(300, 300),  # all images will be resized to 300*300
    batch_size=128,
    class_mode='binary'  # human or horse, so binary
)
print(train_generator)


#train the model
history = model.fit(train_generator, steps_per_epoch=8, epochs=15, verbose=1)


# prepare predict data
def image_to_array(filename):
    img = keras.preprocessing.image.load_img(os.path.join('dataset', filename), target_size=(300, 300))
    x = keras.preprocessing.image.img_to_array(img)
    x /= 255
    return np.expand_dims(x, axis=0)


predict_imgs = np.vstack((image_to_array('predict_horse.jpg'), image_to_array('predict_human.jpg')))
classes = model.predict(predict_imgs, batch_size=10)
print(f'Predict probability: {classes}')
print(f"Readable result: {['Human' if c[0] > 0.5 else 'Horse' for c in classes]}")
