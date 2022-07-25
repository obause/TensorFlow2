from typing import Tuple
import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist


class MNIST:
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Preprocess x data
        self.x_train = x_train.astype(np.float32)
        self.x_train = np.expand_dims(x_train, axis=-1)  # Zusätzliche Dim hinzufügen
        self.x_test = x_test.astype(np.float32)
        self.x_test = np.expand_dims(x_test, axis=-1)  # Zusätzliche Dim hinzufügen
        print(f"x_train shape: {x_train.shape}")

        # Dataset attributes
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
        print(f"train_size:\t{self.train_size}")
        print(f"test_size:\t{self.test_size}")
        print(f"width:\t\t{self.width}")
        print(f"height:\t\t{self.height}")
        print(f"depth:\t\t{self.depth}")
        print(f"img_shape:\t{self.img_shape}")
        print(f"num_classes:\t{self.num_classes}\n")


if __name__ == '__main__':
    data = MNIST()

    data.print_attributes()

    print(f"x_train shape:\t\t{data.x_train.shape}")
    print(f"y_train shape:\t\t{data.y_train.shape}")
    print(f"Max value in x_train:\t{data.x_train.max()}")
    print(f"Min value in x_train:\t{data.x_train.min()}")

    #print(f"x_test shape: {data.x_test.shape}")
    #print(f"y_test shape: {data.y_test.shape}")
