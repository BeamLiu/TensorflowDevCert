import tensorflow as tf
import keras as keras
import numpy as np
import matplotlib.pyplot as plt


class EpochCallbak(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss') < 0.2):
            print(f'loss is low({logs.get("loss")}) and cancel training.')
            self.model.stop_training = True


# load dataset
(training_images, training_labels), (testing_images, testing_labels) = keras.datasets.fashion_mnist.load_data()

# preview the dataset
index = 0
# visualize the image
plt.imshow(training_images[index], cmap='Greys')
plt.show()

# normalize the data
training_images = training_images / 255
testing_images = testing_images / 255

# build the model
model = keras.models.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu,
                        input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(3, 3),
    keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu,
                        input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)])

print(model.summary())
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5, callbacks=[EpochCallbak()])
result = model.evaluate(testing_images, testing_labels)
print(result)

print(testing_labels[:100])
fig, axs = plt.subplots(3, 4)

first_image_index = 0;
second_image_index = 23
third_image_index = 28
convolution_number = 1

layer_output = [layer.output for layer in model.layers]
activation_models = keras.models.Model(inputs=model.inputs, outputs=layer_output)
for x in range(0, 4):
    f1 = activation_models.predict(testing_images[first_image_index].reshape(1, 28, 28, 1))[x]
    axs[0, x].imshow(f1[0, :, :, convolution_number], cmap='inferno')
    axs[0, x].grid(False)

    f2 = activation_models.predict(testing_images[second_image_index].reshape(1, 28, 28, 1))[x]
    axs[1, x].imshow(f2[0, :, :, convolution_number], cmap='inferno')
    axs[1, x].grid(False)

    f3 = activation_models.predict(testing_images[third_image_index].reshape(1, 28, 28, 1))[x]
    axs[2, x].imshow(f3[0, :, :, convolution_number], cmap='inferno')
    axs[2, x].grid(False)

plt.show()
