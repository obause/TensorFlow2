from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import Constant
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def r_squared(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    error = tf.math.subtract(y_true, y_pred)
    sqaured_error = tf.math.square(error)
    numerator = tf.math.reduce_sum(sqaured_error)
    y_true_mean = tf.math.reduce_mean(y_true)
    mean_deviation = tf.math.subtract(y_true, y_true_mean)
    squared_mean_deviation = tf.math.square(mean_deviation)
    denominator = tf.reduce_sum(squared_mean_deviation)
    r2 = tf.math.subtract(1.0, tf.math.divide(numerator, denominator))
    r2_cliped = tf.clip_by_value(r2, clip_value_min=0.0, clip_value_max=1.0)
    return r2_cliped


def get_dataset() -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
]:
    dataset = load_diabetes()
    x = dataset.data
    y = dataset.target.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    return (x_train, y_train), (x_test, y_test)


def build_model(num_features: int, num_targets: int) -> Sequential:
    init_w = RandomUniform(minval=-1.0, maxval=1.0)
    init_b = Constant(value=0.0)

    model = Sequential()
    model.add(
        Dense(
            units=10,
            kernel_initializer=init_w,
            bias_initializer=init_b,
            input_shape=(num_features,),
        )
    )
    model.add(Activation("relu"))
    model.add(
        Dense(
            units=num_targets,
            kernel_initializer=init_w,
            bias_initializer=init_b,
        )
    )
    model.summary()
    return model


def main() -> None:
    (x_train, y_train), (x_test, y_test) = get_dataset()

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    num_features = x_train.shape[1]
    num_targets = y_train.shape[1]

    model = build_model(num_features, num_targets)

    opt = Adam(learning_rate=0.01)

    model.compile(loss="mse", optimizer=opt, metrics=[r_squared])

    model.fit(
        x=x_train,
        y=y_train,
        epochs=3_000,
        batch_size=128,
        verbose=1,
        validation_data=(x_test, y_test),
    )


if __name__ == "__main__":
    main()
