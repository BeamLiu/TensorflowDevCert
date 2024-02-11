import keras
import tensorflow as tf
import keras.datasets.mnist as mnist

print("TensorFlow version:", tf.__version__)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.int8)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128)
print(x_train.shape, y_train.shape, x_train.dtype, y_train.dtype)

lr = 1e-3
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
# (60k, 784) =batch=> (batch_size, 784) =hidden network=> (batch_size, 256) =hidden network=> (batch_size, 128) =output=> (batch_size, 10)
for epoch in range(10):
    for step, (x, y) in enumerate(train_dataset):
        x = tf.reshape(x, shape=(-1, 28 * 28))

        w1 = tf.Variable(tf.random.truncated_normal((784, 256), stddev=0.1))
        b1 = tf.Variable(tf.zeros((256,)))
        w2 = tf.Variable(tf.random.truncated_normal((256, 128), stddev=0.1))
        b2 = tf.Variable(tf.zeros((128,)))
        w3 = tf.Variable(tf.random.truncated_normal((128, 10), stddev=0.1))
        b3 = tf.Variable(tf.zeros((10,)))

        with tf.GradientTape() as tape:
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            output = h2 @ w3 + b3

            # loss = loss_object(y, output)
            # print(loss, y.shape, output.shape, y, output)
            y_onehot = tf.one_hot(y, depth=10)
            loss = tf.reduce_mean(tf.square(y_onehot - output))

        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        if step % 100 == 0:
            print(f'epoch {epoch}, step {step // 100} ==> loss: {loss}')
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])
