import os
from typing import Tuple

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from mnistDataAdvanced import MNIST


def build_model(img_shape: Tuple[int, int, int], num_classes: int) -> Model:
    input_img = Input(shape=img_shape)

    x = Conv2D(filters=32, kernel_size=3, padding="same")(input_img)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=3, padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=64, kernel_size=3, padding="same")(x)
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


if __name__ == "__main__":
    data = MNIST()

    train_dataset = data.get_train_set()
    val_dataset = data.get_val_set()

    model = build_model(data.img_shape, data.num_classes)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0005),
        metrics=["accuracy"]
    )

    model.fit(
        train_dataset,
        epochs=10,
        batch_size=128,
        verbose=1,
        validation_data=(val_dataset)
    )
