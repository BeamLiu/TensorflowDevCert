import csv

import keras
import matplotlib.pyplot as plt
import numpy as np

from Course4.Utility import prepare_train_val_data, window_dataset, show_model_history, plot_series

time_step = []
sunspots = []
with open('./dataset/Sunspots.csv') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader)
    for row in reader:
        time_step.append(row[0])
        sunspots.append(row[2])

time = np.array(time_step, dtype=int)
series = np.array(sunspots, dtype=float)
plt.plot(time, series)
plt.show()

print(f'total data points: {len(time)}')

split_time = 3000
window_size = 60
batch_size = 100
shuffle_buffer_size = 1000
time_train, x_train, time_valid, x_valid = prepare_train_val_data(time, series, split_time)
train_dataset = window_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = keras.models.Sequential([
    keras.layers.Conv1D(filters=32, kernel_size=5,
                           strides=1, padding="causal",
                           activation="relu",
                           input_shape=[None, 1]),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(1),
    keras.layers.Lambda(lambda x: x * 400)
])
model.compile(loss=keras.losses.huber, optimizer=keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9),
              metrics=['mae'])
history = model.fit(train_dataset, epochs=100, verbose=1)
show_model_history(history)


forecast = []
for time in range(split_time - window_size, len(series) - window_size):
    if time % 100 == 0:
        print(time)
    forecast.append(model.predict(series[time:time + window_size][np.newaxis], verbose=0))

# forecast = forecast[split_time - window_size:]
predict_result = np.array(forecast).squeeze()
plot_series(time_valid, x_valid)
plot_series(time_valid, predict_result)
plt.show()
