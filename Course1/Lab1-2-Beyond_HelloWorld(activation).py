import numpy as np
import tensorflow as tf
import keras

print('==========softmax activation(to probability of each element, all the probability sum is 1)===========')
inputs = np.array([[1, 3, 4, 3, 5]], dtype=float)
tensor = tf.convert_to_tensor(inputs)
print(f'data submit to softmax {tensor}')

output = keras.activations.softmax(tensor)
print((f'output of softmax function: {output}'))

# get sum value, should be 1
sum_val = tf.reduce_sum(output)
print(f'sum of the output: {sum_val}')

# max value
prediction = np.argmax(output)
print((f'class with highest probability index: {prediction}, value is {inputs[0][prediction]}'))

print('==========Relu activation(return x if x>0 otherwise 0)===========')
inputs = np.array([[-1, -3, 0, 3, 5]], dtype=float)
tensor = tf.convert_to_tensor(inputs)
print(f'data submit to ReLU {tensor}')

output = keras.activations.relu(tensor)
print((f'output of ReLU function: {output}'))
