import tensorflow as tf
import numpy as np
import keras as keras

print(tf.__version__)

# create the model
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# compile the model
model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.MeanSquaredError())

# prepare data
xs = np.array([-1, 0, 1, 2, 3, 4], dtype=float)
ys = np.array([-3, -1, 1, 3, 5, 7], dtype=float)

# train the neural network
model.fit(x=xs, y=ys, epochs=500)

# predict y for given x=10
print(model.predict(x=[10, 20, 30]))
