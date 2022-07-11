from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


np.random.seed(0)
tf.random.set_seed(0)


class MNIST:
    def __init__(self, with_normalization: bool = True) -> None:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x_train_: np.ndarray = None
        self.y_train_: np.ndarray = None
        self.x_val_: np.ndarray = None
        self.y_val_: np.ndarray = None
        self.val_size = 0
        self.train_splitted_size = 0
        # Preprocess x data
        self.x_train = x_train.astype(np.float32)
        self.x_train = np.expand_dims(x_train, axis=-1)
        if with_normalization:
            self.x_train = self.x_train / 255.0
        self.x_test = x_test.astype(np.float32)
        self.x_test = np.expand_dims(x_test, axis=-1)
        if with_normalization:
            self.x_test = self.x_test / 255.0
        # Dataset attributes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.depth = self.x_train.shape[3]
        self.img_shape = (self.width, self.height, self.depth)
        self.num_classes = 10
        # Preprocess y data
        self.y_train = to_categorical(y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(y_test, num_classes=self.num_classes)

    def get_train_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_train, self.y_train

    def get_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_test, self.y_test

    def get_splitted_train_validation_set(self, validation_size: float = 0.33) -> tuple:
        (
            self.x_train_,
            self.x_val_,
            self.y_train_,
            self.y_val_,
        ) = train_test_split(self.x_train, self.y_train, test_size=validation_size)
        self.val_size = self.x_val_.shape[0]
        self.train_splitted_size = self.x_train_.shape[0]
        return self.x_train_, self.x_val_, self.y_train_, self.y_val_

    def data_augmentation(self, augment_size: int = 5_000) -> None:
        image_generator = ImageDataGenerator(
            rotation_range=5,
            zoom_range=0.08,
            width_shift_range=0.08,
            height_shift_range=0.08,
        )
        # Fit the data generator
        image_generator.fit(self.x_train, augment=True)
        # Get random train images for the data augmentation
        rand_idxs = np.random.randint(self.train_size, size=augment_size)
        x_augmented = self.x_train[rand_idxs].copy()
        y_augmented = self.y_train[rand_idxs].copy()
        x_augmented = image_generator.flow(
            x_augmented,
            np.zeros(augment_size),
            batch_size=augment_size,
            shuffle=False,
        ).next()[0]
        # Append the augmented images to the train set
        self.x_train = np.concatenate((self.x_train, x_augmented))
        self.y_train = np.concatenate((self.y_train, y_augmented))
        self.train_size = self.x_train.shape[0]
