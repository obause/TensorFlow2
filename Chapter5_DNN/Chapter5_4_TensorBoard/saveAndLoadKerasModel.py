from typing import Tuple

import numpy as np
import os

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model

MODELS_DIR = os.path.join(os.path.curdir, "/models")
MODEL_FILE_PATH = os.path.join(MODELS_DIR, "/mnist_model.h5")
FULL_MODEL_FILE_PATH = os.path.join(MODELS_DIR, "full_mnist_model", "full_mnist_model.h5")
print(f"Model directory: {MODELS_DIR}")


def get_dataset(num_features: int, num_classes: int) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray]]:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, num_features).astype(np.float32)
    x_test = x_test.reshape(-1, num_features).astype(np.float32)
    y_train = to_categorical(y_train, num_classes=num_classes, dtype=np.float32)
    y_test = to_categorical(y_test, num_classes=num_classes, dtype=np.float32)

    return (x_train, y_train), (x_test, y_test)


def build_model(num_features: int, num_classes: int) -> Sequential:
    """ Returns a Sequential model. """
    init_weights = RandomUniform(minval=-0.05, maxval=0.05)
    init_bias = Constant(value=0.0)
    model = Sequential()
    model.add(
        Dense(
            units=256,
            kernel_initializer=init_weights,
            bias_initializer=init_bias,
            input_shape=(num_features,)
        )
    )
    model.add(Activation('relu'))
    model.add(
        Dense(
            units=128,
            kernel_initializer=init_weights,
            bias_initializer=init_bias,
            input_shape=(num_features,)
        )
    )
    model.add(Activation('relu'))
    model.add(
        Dense(
            units=64,
            kernel_initializer=init_weights,
            bias_initializer=init_bias,
            input_shape=(num_features,)
        )
    )
    model.add(Activation('relu'))
    model.add(Dense(units=num_classes))
    model.add(Activation('softmax'))
    model.summary()
    return model


def main():
    """ Main function. """
    num_features = 784
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = get_dataset(num_features, num_classes)

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # model = build_model(
    #     num_features=num_features,
    #     num_classes=num_classes
    # )

    # optimizer = Adam(lr=0.001)
    # # optimizer = SGD()
    # # optimizer = RMSprop()

    # model.compile(
    #     loss='categorical_crossentropy',
    #     optimizer=optimizer,
    #     metrics=["accuracy"]
    # )

    # model.fit(
    #     x=x_train,
    #     y=y_train,
    #     epochs=50,
    #     batch_size=128,
    #     verbose=1,
    #     validation_data=(x_test, y_test)
    # )

    # scores = model.evaluate(x_test, y_test)
    # print(f"Scores before saving: {scores}")

    # Save the model to a file.
    # model.save_weights(filepath=MODEL_FILE_PATH)
    # model.load_weights(filepath=MODEL_FILE_PATH)
    # model.save(filepath=FULL_MODEL_FILE_PATH)

    # Load the model from a file.
    model = load_model(filepath=FULL_MODEL_FILE_PATH)
    model.summary()

    scores = model.evaluate(x_test, y_test)
    print(f"Scores after loading: {scores}")


if __name__ == '__main__':
    main()
