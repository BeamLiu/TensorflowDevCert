import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.src.callbacks import History


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.1,
                    np.cos(season_time * 7 * np.pi),
                    1 / np.exp(5 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def prepare_timeseries_data():
    plt.figure(figsize=(10, 6))
    time = np.arange(4 * 365 + 1, dtype="float32")
    series = trend(time, 0.1)
    plot_series(time, series)
    baseline = 10
    amplitude = 40
    slope = 0.01
    noise_level = 2

    # Create the series
    series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
    # Update with noise
    series += noise(time, noise_level, seed=42)
    plot_series(time, series)
    plt.show()

    return time, series


def prepare_timeseries_data():
    plt.figure(figsize=(10, 6))
    time = np.arange(4 * 365 + 1, dtype="float32")
    series = trend(time, 0.1)
    plot_series(time, series)
    plt.show()
    baseline = 10
    amplitude = 40
    slope = 0.01
    noise_level = 2

    # Create the series
    series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
    # Update with noise
    series += noise(time, noise_level, seed=42)
    plot_series(time, series)
    plt.show()

    return time, series


def prepare_train_val_data(time, series, split_time):
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]
    plt.figure(figsize=(10, 6))
    plot_series(time_train, x_train)
    plt.show()

    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid)
    plt.show()
    return time_train, x_train, time_valid, x_valid


def window_dataset(series, window_size, batch_size, shuffle_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset.shuffle(shuffle_size)
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def show_model_history(history: History, name=None):
    # Extracting accuracy and loss values from history
    mae = history.history['mae']
    loss = history.history['loss']
    has_val = 'val_mae' in history.history
    if has_val:
        val_mae = history.history['val_mae']
        val_loss = history.history['val_loss']
    epochs = range(len(mae))

    # Display accuracy and loss in two columns
    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, mae, label='Train MAE')
    if has_val:
        plt.plot(epochs, val_mae, label='Validation MAE')
        plt.title(f'Train vs. Validation MAE {"(" + name + ")" if name is not None else ""}')
    else:
        plt.title(f'Train MAE {"(" + name + ")" if name is not None else ""}')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend(loc='upper right')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss')
    if has_val:
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.title(f'Train vs. Validation Loss {"(" + name + ")" if name is not None else ""}')
    else:
        plt.title(f'Train Loss {"(" + name + ")" if name is not None else ""}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
