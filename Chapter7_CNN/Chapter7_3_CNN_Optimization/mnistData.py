from typing import Tuple
import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split


class MNIST:
    def __init__(self, normalize: bool = True) -> None:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        self.x_train_: np.ndarray = None
        self.y_train_: np.ndarray = None
        self.x_val_: np.ndarray = None
        self.y_val_: np.ndarray = None
        self.val_size: int = 0
        self.train_splitted_size: int = 0

        # Preprocess x data
        self.x_train = x_train.astype(np.float32)
        self.x_train = np.expand_dims(x_train, axis=-1)  # Zusätzliche Dim hinzufügen
        self.x_test = x_test.astype(np.float32)
        self.x_test = np.expand_dims(x_test, axis=-1)  # Zusätzliche Dim hinzufügen
        print(f"x_train shape: {x_train.shape}")

        # One-Hot-Encoding
        if normalize:
            self.x_train = self.x_train / 255.0
            self.x_test = self.x_test / 255.0

        # Dataset attributes
        self.normalize = normalize
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.depth = self.x_train.shape[3]
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.img_shape = (self.width, self.height, self.depth)
        self.num_classes = 10
        # self.num_classes = len(np.unique(self.y_train))

        # Preprocess y data
        self.y_train = to_categorical(y_train, num_classes=self.num_classes, dtype=np.float32)
        self.y_test = to_categorical(y_test, num_classes=self.num_classes, dtype=np.float32)

    def get_train_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_train, self.y_train

    def get_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_test, self.y_test

    def print_attributes(self):
        print("\nDataset attributes:")
        print(f"Normalize: {self.normalize}")
        print(f"train_size:\t{self.train_size}")
        print(f"test_size:\t{self.test_size}")
        print(f"width:\t\t{self.width}")
        print(f"height:\t\t{self.height}")
        print(f"depth:\t\t{self.depth}")
        print(f"img_shape:\t{self.img_shape}")
        print(f"num_classes:\t{self.num_classes}\n")

    def get_train_val_set(self, val_size: float = 0.33) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Split train set into train and validation set. """
        self.x_train_, self.x_val_, self.y_train_, self.y_val_ = train_test_split(
            self.x_train,
            self.y_train,
            test_size=val_size,
            random_state=42
        )
        self.val_size = self.x_val_.shape[0]
        self.train_splitted_size = self.x_train_.shape[0]
        return self.x_train_, self.x_val_, self.y_train_, self.y_val_

    def data_augmentation(self, augment_size: int = 5000) -> None:
        image_generator = ImageDataGenerator(
            rotation_range=5,
            zoom_range=0.08,
            width_shift_range=0.08,
            height_shift_range=0.08,
        )

        # Fit the data generator
        image_generator.fit(self.x_train, augment=True)

        # Get random train images for augmentation
        rand_idxs = np.random.randint(self.train_size, size=augment_size)
        x_augmented = self.x_train[rand_idxs].copy()
        y_augmented = self.y_train[rand_idxs].copy()
        plt.imshow(x_augmented[0, :, :, 0], cmap="gray")
        plt.show()
        x_augmented = image_generator.flow(
            x_augmented,
            np.zeros(augment_size),
            batch_size=augment_size,
            shuffle=False
        ).next()[0]
        plt.imshow(x_augmented[0, :, :, 0], cmap="gray")
        plt.show()

        # Append augmented images to train set
        self.x_train = np.concatenate((self.x_train, x_augmented))
        self.y_train = np.concatenate((self.y_train, y_augmented))
        self.train_size = self.x_train.shape[0]


if __name__ == '__main__':
    data = MNIST()

    data.print_attributes()

    print(f"x_train shape:\t\t{data.x_train.shape}")
    print(f"y_train shape:\t\t{data.y_train.shape}")

    print(f"Max value in x_train:\t{data.x_train.max()}")
    print(f"Min value in x_train:\t{data.x_train.min()}")

    data.data_augmentation(augment_size=5000)

    print(f"Augmented x_train shape:\t\t{data.x_train.shape}")
    print(f"Augmented y_train shape:\t\t{data.y_train.shape}")

    # print(f"x_test shape: {data.x_test.shape}")
    # print(f"y_test shape: {data.y_test.shape}")
