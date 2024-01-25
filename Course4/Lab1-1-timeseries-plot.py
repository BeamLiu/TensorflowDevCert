import keras
import matplotlib.pyplot as plt
import numpy as np

from Course4.Utility import trend, plot_series, seasonality, noise


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


def prepare_train_val_data(split_time):
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


time, series = prepare_timeseries_data()

split_time = 1100
time_train, x_train, time_valid, x_valid = prepare_train_val_data(split_time)
# navie forecast
naive_forecast = series[split_time - 1:-1]
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, naive_forecast)
plt.show()

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, start=0, end=150)
plot_series(time_valid, naive_forecast, start=1, end=151)
print(keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())
plt.show()


def moving_average_forecast(series, window_size):
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    return np.array(forecast)

moving_avg = moving_average_forecast(series, 30)[split_time - 30:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, moving_avg)
print(keras.metrics.mean_squared_error(x_valid, moving_avg).numpy())
print(keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy())
plt.show()
