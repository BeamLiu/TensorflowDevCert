import keras
import tensorflow as tf
import keras.datasets.mnist as mnist

print("TensorFlow version:", tf.__version__)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
print(f'predictions: {predictions}')
#The tf.nn.softmax function converts these logits to probabilities for each class
print(f'softmax prediction: {tf.nn.softmax(predictions).numpy()}')
#Define a loss function for training using losses.SparseCategoricalCrossentropy:
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to -tf.math.log(1/10) ~= 2.3
print(f'loss fn: {loss_fn(y_train[:1], predictions).numpy()}')

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

#If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it:
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

print(f'probability: {probability_model(x_test[:5])}')
