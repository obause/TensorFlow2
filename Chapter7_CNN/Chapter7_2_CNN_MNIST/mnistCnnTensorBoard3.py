from typing import Tuple

import numpy as np
import os

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import TensorBoard

LOGS_DIR = os.path.join(os.path.curdir, "logs")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)
MODEL_LOG_DIR = os.path.join(LOGS_DIR, "mnist_cnn3")
print(f"Log directory: {LOGS_DIR}")


def prepare_dataset(num_classes: int) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray]]:

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype(np.float32)
    x_train = np.expand_dims(x_train, axis=-1)  # Zusätzliche Dim hinzufügen
    x_test = x_test.astype(np.float32)
    x_test = np.expand_dims(x_test, axis=-1)  # Zusätzliche Dim hinzufügen
    print(f"x_train shape: {x_train.shape}")

    y_train = to_categorical(y_train, num_classes=num_classes, dtype=np.float32)
    y_test = to_categorical(y_test, num_classes=num_classes, dtype=np.float32)

    return (x_train, y_train), (x_test, y_test)


def build_model(img_shape: Tuple[int, int, int], num_classes: int) -> Sequential:
    """ Returns a Sequential model. """

    # Sequenzielles
    model = Sequential()

    # Convolutional layer 1
    model.add(Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        input_shape=img_shape
    ))
    model.add(Activation("relu"))
    model.add(MaxPooling2D())

    # Convolutional layer 2
    model.add(Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        input_shape=img_shape
    ))
    model.add(Activation("relu"))
    model.add(MaxPooling2D())

    # Convolutional layer 2
    model.add(Conv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        input_shape=img_shape
    ))
    model.add(Activation("relu"))
    model.add(MaxPooling2D())

    # Nach jedem Conv2D-Layer wird ein Reshape-Layer eingefügt.
    # Dieser Reshape-Layer ist notwendig, um die Dimensionen der
    # Input- und Output-Variablen zu ändern.
    # model.add(Reshape(target_shape=))
    model.add(Flatten())

    model.add(Dense(units=num_classes))
    model.add(Activation("softmax"))

    model.summary()
    return model


def main():
    """ Main function. """
    img_shape = (28, 28, 1)
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = prepare_dataset(num_classes)

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    model = build_model(
        img_shape=img_shape,
        num_classes=num_classes
    )

    optimizer = Adam(lr=0.001)
    # optimizer = SGD()
    # optimizer = RMSprop()

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    tb_callback = TensorBoard(
        log_dir=MODEL_LOG_DIR,
        histogram_freq=1,
        write_graph=True
    )

    model.fit(
        x=x_train,
        y=y_train,
        epochs=25,
        batch_size=128,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[tb_callback]
    )

    scores = model.evaluate(x_test, y_test)
    print(f"Scores: {scores}")


if __name__ == '__main__':
    main()
