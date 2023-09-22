import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

import datasets_path

import os
import sys


def get_train_and_test(path_to_dataset_location):
    train_localization = Rf"{path_to_dataset_location}\10_food_classes_all_data\train"
    test_localization = Rf"{path_to_dataset_location}\10_food_classes_all_data\test"
    train_generator = ImageDataGenerator(
        rescale=1.0 / 255.0,
        channel_shift_range=0.2,
        height_shift_range=0.2,
        width_shift_range=0.2,
        rotation_range=20,
        horizontal_flip=True,
        validation_split=0.1,
        zoom_range=0.2,
    )

    test_generator = ImageDataGenerator(rescale=1.0 / 255.0)

    test = test_generator.flow_from_directory(
        directory=test_localization,
        target_size=(256, 256),
        batch_size=32,
        shuffle=True,
        class_mode="sparse",
    )

    train = train_generator.flow_from_directory(
        directory=train_localization,
        target_size=(256, 256),
        batch_size=32,
        shuffle=True,
        class_mode="sparse",
        subset="training",
    )

    validation = train_generator.flow_from_directory(
        directory=train_localization,
        target_size=(256, 256),
        batch_size=32,
        shuffle=True,
        class_mode="sparse",
        subset="validation",
    )

    return (train, validation, test)


def study_tendency(history):
    train_loss = history["loss"]
    train_accuracy = history["sparse_categorical_accuracy"]

    val_loss = history["val_loss"]
    val_accuracy = history["val_sparse_categorical_accuracy"]

    epochs = range(len(history["loss"]))

    plt.figure()
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, val_loss, label="val loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.figure()
    plt.plot(epochs, train_accuracy, label="train accurancy")
    plt.plot(epochs, val_accuracy, label="val accurancy")
    plt.title("Accurancy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


def main():
    train, validation, test = get_train_and_test(datasets_path.get_datasets_path())

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Resizing(256, 256),
            tf.keras.layers.Conv2D(10, 3, activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2D(10, 3, activation=tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(10, 3, activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2D(10, 3, activation=tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax),
        ]
    )

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    history = model.fit(
        train,
        steps_per_epoch=len(train),
        verbose=1,
        epochs=8,
        validation_data=validation,
        validation_steps=len(validation),
    )

    study_tendency(history.history)

    model.evaluate(test, steps=len(test), verbose=1)


if __name__ == "__main__":
    is_conda = os.path.exists(os.path.join(sys.prefix, "conda-meta", "history"))
    if not is_conda:
        raise EnvironmentError()
    main()
