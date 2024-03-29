# Dataset: https://www.microsoft.com/en-us/download/details.aspx?id=54765
import os
import cv2

import numpy as np
import tensorflow as tf

from typing import Tuple
from skimage import transform

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


np.random.seed(0)
tf.random.set_seed(0)

# DATA_DIR = "C:/Users/Ole/Nextcloud/Uni/Kurse/0Datasets/kagglecatsanddogs_5340/PetImages"
DATA_DIR = "C:/Users/oleba/Documents/datasets/PetImages"
X_FILE_PATH = os.path.join(DATA_DIR, "x.npy")
Y_FILE_PATH = os.path.join(DATA_DIR, "y.npy")
IMG_SIZE = 64
IMG_DEPTH = 3
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_DEPTH)


def extract_cats_dogs() -> None:
    """ Extracts the cats and dogs images from the dataset and saves them as numpy arrays. """
    cats_dir = os.path.join(DATA_DIR, "Cat").replace("\\","/")
    dogs_dir = os.path.join(DATA_DIR, "Dog").replace("\\","/")

    dirs = [cats_dir, dogs_dir]
    class_names = ["cat", "dog"]

    # Falsche Dateien löschen
    for d in dirs:
        for f in os.listdir(d):
            if not f.endswith(".jpg"):
                print(f"Removing file {f}")
                os.remove(os.path.join(d, f))

    num_cats = len(os.listdir(cats_dir))
    num_dogs = len(os.listdir(dogs_dir))
    num_images = num_cats + num_dogs
    print(f"Found {num_cats} cats and {num_dogs} dogs.")
    x = np.zeros((num_images, IMG_SIZE, IMG_SIZE, IMG_DEPTH), dtype=np.float32)
    y = np.zeros((num_images,), dtype=np.int32)

    count = 0
    for d, class_name in zip(dirs, class_names):
        for f in os.listdir(d):
            #print(f"Processing file {f}")
            img_filename = os.path.join(d, f).replace("\\","/")
            try:
                img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x[count] = transform.resize(
                    image=img,
                    output_shape=IMG_SHAPE
                )
                if class_name == "cat":
                    y[count] = 0
                elif class_name == "dog":
                    y[count] = 1
                else:
                    print(f"Unknown class name {class_name}")
                count += 1
            except Exception as e:
                print(f"Error processing {img_filename}: {e}")

    x = x[:count]
    y = y[:count]

    np.save(X_FILE_PATH, x)
    np.save(Y_FILE_PATH, y)


def extract_cats_dogs2() -> None:
    x = []
    y = []
    for folder in ["Cat", "Dog"]:
        for file in os.listdir(os.path.join(DATA_DIR, folder)):
            if file.endswith(".jpg"):
                img = cv2.imread(os.path.join(DATA_DIR, folder, file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                x.append(img)
                y.append(0 if folder == "Cat" else 1)
    x = np.array(x)
    y = np.array(y)
    np.save(X_FILE_PATH, x)
    np.save(Y_FILE_PATH, y)


class DOGSCATS:
    def __init__(
        self, test_size: float = 0.2, validation_size: float = 0.33
    ) -> None:
        # User-definen constants
        self.num_classes = 10
        self.batch_size = 128
        # Load the data set
        x = np.load(X_FILE_PATH)
        y = np.load(Y_FILE_PATH)
        # Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=validation_size
        )
        # Preprocess x data
        self.x_train = x_train.astype(np.float32)
        self.x_test = x_test.astype(np.float32)
        self.x_val = x_val.astype(np.float32)
        # Preprocess y data
        self.y_train = to_categorical(y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(y_test, num_classes=self.num_classes)
        self.y_val = to_categorical(y_val, num_classes=self.num_classes)
        # Dataset attributes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.val_size = self.x_val.shape[0]
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.depth = self.x_train.shape[3]
        self.img_shape = (self.width, self.height, self.depth)

    def get_train_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_train, self.y_train

    def get_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_test, self.y_test

    def get_val_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_val, self.y_val

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


if __name__ == "__main__":
    # extract_cats_dogs()

    data = DOGSCATS()
