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
    if not os.path.exists(f'{target_folder}/cats_and_dogs_filtered') or not any(
            os.listdir(f'{target_folder}/cats_and_dogs_filtered')):
        print('extracting the images')
        zip_ref = zipfile.ZipFile(zip_file)
        zip_ref.extractall(target_folder)
        zip_ref.close()

    target_folder = os.path.join(target_folder, 'cats_and_dogs_filtered')
    print(f'the images {zip_file} are prepared!')
    train_cat_dir = os.path.join(f'{target_folder}/train/cats')
    train_dog_dir = os.path.join(f'{target_folder}/train/dogs')
    val_cat_dir = os.path.join(f'{target_folder}/validation/cats')
    val_dog_dir = os.path.join(f'{target_folder}/validation/dogs')
    train_cat_files = [os.path.join(train_cat_dir, file) for file in os.listdir(train_cat_dir) if
                       os.path.isfile(os.path.join(train_cat_dir, file))]
    train_dog_files = [os.path.join(train_dog_dir, file) for file in os.listdir(train_dog_dir) if
                       os.path.isfile(os.path.join(train_dog_dir, file))]
    val_cat_files = [os.path.join(val_cat_dir, file) for file in os.listdir(val_cat_dir) if
                     os.path.isfile(os.path.join(val_cat_dir, file))]
    val_dog_files = [os.path.join(val_dog_dir, file) for file in os.listdir(val_dog_dir) if
                     os.path.isfile(os.path.join(val_dog_dir, file))]
    print(f'total train cat {train_cat_dir} images {len(train_cat_files)}')
    print(f'total train dog {train_dog_dir} images {len(train_dog_files)}')
    print(f'total val cat {val_cat_dir} images {len(val_cat_files)}')
    print(f'total val dog {val_dog_dir} images {len(val_dog_files)}')
    return train_cat_files, train_dog_files, val_cat_files, val_dog_files


def image_to_array(filename: str):
    img = load_img(os.path.join(filename), target_size=(150, 150))
    x = keras.preprocessing.image.img_to_array(img)
    x /= 255
    return x


def random_display_images(dog_files: array, cat_files: array):
    global f, axs, i
    # list random 8 images for each
    f, axs = plt.subplots(2, 8, figsize=(32, 12))
    for (i, horse_file, human_file) in zip(range(0, 8), np.random.choice(dog_files, size=8),
                                           np.random.choice(cat_files, size=8)):
        axs[0, i].axis('off')
        axs[1, i].axis('off')
        axs[0, i].imshow(matplotlib.image.imread(horse_file))
        axs[1, i].imshow(matplotlib.image.imread(human_file))
    plt.show()


def show_model_history(history: History, name=None):
    # Extracting accuracy and loss values from history
    acc = history.history['accuracy']
    loss = history.history['loss']
    has_val = 'val_accuracy' in history.history
    if has_val:
        val_acc = history.history['val_accuracy']
        val_loss = history.history['val_loss']
    epochs = range(len(acc))

    # Display accuracy and loss in two columns
    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Accuracy')
    if has_val:
        plt.plot(epochs, val_acc, label='Validation Accuracy')
        plt.title(f'Train vs. Validation Accuracy {"(" + name + ")" if name is not None else ""}')
    else:
        plt.title(f'Train Accuracy {"(" + name + ")" if name is not None else ""}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
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


def visualize_layers(model, target_img, max_features_in_cnn_layers):
    successive_outputs = [layer.output for layer in model.layers]
    visualization_model = keras.models.Model(inputs=model.inputs, outputs=successive_outputs)
    img = image_to_array(target_img)
    plt.imshow(img)
    plt.title('image used to visualize CNN')
    plt.show()
    successive_feature_maps = visualization_model.predict(np.array([img]))
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
