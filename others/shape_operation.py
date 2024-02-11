import numpy as np
import tensorflow as tf


def tf_operation():
    a = tf.random.normal((4, 3, 2, 1))
    assert a.shape == (4, 3, 2, 1)

    a = tf.reshape(tf.range(0, 100), (2, 5, 5, 2))
    assert a.shape == (2, 5, 5, 2)

    a = tf.reshape(a, (2, 25, 2))

    a = tf.reshape(a, (4, 25))
    assert len(a[0]) == 25
    assert a[1][0] == 25

    a = tf.expand_dims(a, axis=-1)
    assert a.shape == (4, 25, 1)

    a = tf.expand_dims(a, axis=1)
    assert a.shape == (4, 1, 25, 1)

    a = tf.squeeze(a, axis=-1)
    assert a.shape == (4, 1, 25)

    a = tf.transpose(a, perm=[2, 0, 1])
    assert a.shape == (25, 4, 1)
    assert np.all(a[0, :, 0] == [0, 25, 50, 75])

    a = tf.transpose(a, perm=[1, 2, 0])
    assert a.shape == (4, 1, 25)

    a = a[..., 0]
    assert a.shape == (4, 1)

    a = tf.squeeze(a, axis=-1)
    assert a.shape == (4,)

    assert np.all(a.numpy() == [0, 25, 50, 75])

    # broadcasting
    a = tf.reshape(tf.range(0, 10.), (2, 5))
    b = tf.ones((5,))
    assert a.shape == (2, 5)
    assert b.shape == (5,)
    # auto broadcasted to same shape, then do add operation
    c = a + b
    assert c.shape == (2, 5)
    assert np.all(c[0].numpy() == [1., 2., 3., 4., 5.])


def np_operation():
    a = np.random.normal(size=(4, 3, 2, 1))
    assert a.shape == (4, 3, 2, 1)

    a = np.reshape(range(0, 100), (2, 5, 5, 2))
    assert a.shape == (2, 5, 5, 2)

    a = np.reshape(a, (2, 25, 2))

    a = np.reshape(a, (4, 25))
    assert len(a[0]) == 25
    assert a[1][0] == 25

    a = np.expand_dims(a, axis=-1)
    assert a.shape == (4, 25, 1)

    a = np.expand_dims(a, axis=1)
    assert a.shape == (4, 1, 25, 1)

    a = np.squeeze(a, axis=-1)
    assert a.shape == (4, 1, 25)

    a = np.transpose(a, axes=[2, 0, 1])
    assert a.shape == (25, 4, 1)
    assert np.all(a[0, :, 0] == [0, 25, 50, 75])

    a = np.transpose(a, axes=[1, 2, 0])
    assert a.shape == (4, 1, 25)

    a = a[..., 0]
    assert a.shape == (4, 1)

    a = np.squeeze(a, axis=-1)
    assert a.shape == (4,)

    assert np.all(a == [0, 25, 50, 75])

    # broadcasting
    a = np.reshape(range(0, 10), (2, 5))
    b = np.ones((5,))
    assert a.shape == (2, 5)
    assert b.shape == (5,)
    # auto broadcasted to same shape, then do add operation
    c = a + b
    assert c.shape == (2, 5)
    assert np.all(c[0] == [1., 2., 3., 4., 5.])


tf_operation()
np_operation()
