import tensorflow as tf
import keras as keras
import numpy as np
import matplotlib.pyplot as plt


class EpochCallbak(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss') < 0.4):
            print(f'loss is low({logs.get("loss")}) and cancel training.')
            self.model.stop_training = True


# load dataset
(training_images, training_labels), (testing_images, testing_labels) = keras.datasets.fashion_mnist.load_data()

# preview the dataset
index = 0
# 320 characters per line
np.set_printoptions(linewidth=320)
print(f'Lable: {training_labels[index]}')
print(f'Image: {training_images[index]}')
# visualize the image
plt.imshow(training_images[index], cmap='Greys')
plt.show()

# normalize the data
training_images = training_images / 255
testing_images = testing_images / 255

# build the model
# Flatten is used to flatten data to one dimension array
# relu:   return x if x>0 otherwise 0
# softmax: to probability of each element, all the probability sum is 1
model = keras.models.Sequential([keras.layers.Flatten(),
                                 keras.layers.Dense(300, activation=tf.nn.relu),
                                 keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=3, callbacks=[EpochCallbak()])
result = model.evaluate(testing_images, testing_labels)
print(result)
