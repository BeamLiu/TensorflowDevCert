import array
import os
import zipfile

import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import load_img
from keras.src.callbacks import History


def prepare_images_from_zip(zip_file: str, target_folder: str):
    if not os.path.exists(target_folder) or not any(os.listdir(target_folder)):
        print('extracting the images')
        zip_ref = zipfile.ZipFile(zip_file)
        zip_ref.extractall(target_folder)
        zip_ref.close()

    print(f'the images {zip_file} are prepared!')
    train_horse_dir = os.path.join(f'{target_folder}/horses')
    train_human_dir = os.path.join(f'{target_folder}/humans')
    horse_files = [os.path.join(train_horse_dir, file) for file in os.listdir(train_horse_dir) if
                   os.path.isfile(os.path.join(train_horse_dir, file))]
    human_files = [os.path.join(train_human_dir, file) for file in os.listdir(train_human_dir) if
                   os.path.isfile(os.path.join(train_human_dir, file))]
    print(f'total horse {train_horse_dir} images {len(horse_files)}')
    print(f'total human {train_human_dir} images {len(human_files)}')
    return horse_files, human_files


def image_to_array(filename: str):
    img = load_img(os.path.join(filename), target_size=(300, 300))
    x = keras.preprocessing.image.img_to_array(img)
    x /= 255
    return np.expand_dims(x, axis=0)


def random_display_images(horse_files: array, human_files: array):
    global f, axs, i
    # list random 8 images for each
    f, axs = plt.subplots(2, 8, figsize=(32, 12))
    for (i, horse_file, human_file) in zip(range(0, 8), np.random.choice(horse_files, size=8),
                                           np.random.choice(human_files, size=8)):
        axs[0, i].axis('off')
        axs[1, i].axis('off')
        axs[0, i].imshow(matplotlib.image.imread(horse_file))
        axs[1, i].imshow(matplotlib.image.imread(human_file))
    plt.show()


def show_model_history(history: History):
    acc = history.history['accuracy']
    loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    # display the  accuracy
    plt.plot(epochs, acc, label='train acc')
    plt.plot(epochs, val_acc, label='validation acc')
    plt.title('Train v.s. Validation accuracy')
    plt.legend(loc='upper right')
    plt.figure()
    plt.show()
    # display the  loss
    plt.plot(epochs, loss, label='train loss')
    plt.plot(epochs, val_loss, label='validation loss')
    plt.title('Train v.s. Validation loss')
    plt.legend(loc='upper right')
    plt.figure()
    plt.show()


def visualize_layers(model, target_img, max_features_in_cnn_layers):
    successive_outputs = [layer.output for layer in model.layers]
    visualization_model = keras.models.Model(inputs=model.inputs, outputs=successive_outputs)
    img = image_to_array(target_img)
    plt.imshow(img[0, :, :, :])
    plt.title('image used to visualize CNN')
    plt.show()
    successive_feature_maps = visualization_model.predict(img)
    layer_names = [layer.name for layer in model.layers]
    f, axs = plt.subplots(len(layer_names) - 3, max_features_in_cnn_layers,
                          figsize=(100, len(layer_names)))  # 3 DNN layers won;'t be painted
    for layer_idx, layer_name, feature_map in zip(range(len(layer_names)), layer_names, successive_feature_maps):
        if len(feature_map.shape) == 4:
            # just do this for the conv/max pool layers, not the fully-connected layers
            # as the fully-connected layers did a flatten action to 3 dimensions
            n_features = feature_map.shape[-1]  # number of features in feature map
            # feature map has shape (1,size,size,n_features)
            size = feature_map.shape[1]
            for i in range(n_features):
                ax = axs[layer_idx, i]
                ax.grid(False)
                ax.axis('off')

                x = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                # show the feature into grid
                axs[layer_idx, i].imshow(x, aspect='auto', cmap='viridis')
                # Add layer_name as title for each row
                if i == 0:
                    ax.set_title(f'{layer_name}({n_features} features)')
    plt.tight_layout()
    plt.show()
