from typing import Tuple

import numpy as np
import os

from matplotlib import pyplot as plt

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import TensorBoard

from Chapter7_CNN.Chapter7_3_CNN_Optimization.mnistData import MNIST


def build_model(img_shape: Tuple[int, int, int], num_classes: int) -> Model:
    """ Builds a CNN model. """

    # Input-Layer
    input_img = Input(shape=img_shape)  # 28x28x1

    # Alternative:
    # conv_layer_object = Conv2D(filters=32, kernel_size=3, padding="same")
    # conv_layer_object(input_img) # Dem Conv2D-Layer ein Input-Objekt zuweisen

    x = Conv2D(filters=32, kernel_size=3, padding="same")(input_img)  # (28, 28, 64)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=3, padding="same")(x)  # (28, 28, 64)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)  # (14, 14, 64)

    x = Conv2D(filters=64, kernel_size=3, padding="same")(input_img)
    x = Activation("relu")(x)
    x = Conv2D(filters=64, kernel_size=3, padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation("softmax")(x)

    model = Model(
        inputs=[input_img],
        outputs=[y_pred]
    )
    model.summary()
    return model


def plot_filters(model: Model) -> None:
    """ Plots the filters of the model. """

    # Get the first conv layer
    conv_layer = model.layers[1]

    # Get the weights of the first convolutional layer.
    weights = conv_layer.get_weights()[0]
    print(f"weights shape: {weights.shape}")

    # Get the number of filters.
    num_filters = weights.shape[3]
    print(f"num_filters: {num_filters}")

    # Create a figure with a grid of size num_rows x num_cols.
    subplot_grid = (num_filters // 4, 4)
    fig, axes = plt.subplots(subplot_grid[0], subplot_grid[1], figsize=(20, 20))
    axes = axes.reshape(num_filters)

    for filter_idx in range(num_filters):
        axes[filter_idx].imshow(weights[:, :, 0, filter_idx], cmap="gray")

    axes = axes.reshape(subplot_grid)
    fig.subplots_adjust(hspace=0.5)
    plt.show()


def main():
    """ Main function. """
    data = MNIST(normalize=False)

    x_train, y_train = data.get_train_set()
    x_test, y_test = data.get_test_set()

    model = build_model(
        img_shape=data.img_shape,
        num_classes=data.num_classes
    )

    optimizer = Adam(lr=0.001)
    # optimizer = SGD()
    # optimizer = RMSprop()

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    model.fit(
        x=x_train,
        y=y_train,
        epochs=1,
        batch_size=128,
        verbose=1,
        validation_data=(x_test, y_test)
    )

    scores = model.evaluate(x_test, y_test)
    print(f"Scores: {scores}")


if __name__ == '__main__':
    main()
