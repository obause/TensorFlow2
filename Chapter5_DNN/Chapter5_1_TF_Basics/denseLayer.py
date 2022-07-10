from typing import Tuple
import tensorflow as tf


def get_dataset() -> Tuple[tf.Tensor, tf.Tensor]:
    """ OR Dataset """
    x = tf.constant([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=tf.float32)
    y = tf.constant([[0], [1], [1], [1]], dtype=tf.float32)
    return x, y


def dense(W: tf.Tensor, b: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    """ dense """
    return tf.add(tf.linalg.matmul(x, W), b)
    # return tf.add(tf.matmul(x, W), b)


def main() -> None:
    x, y = get_dataset()
    print(f"x_shape: {x.shape}")
    print(f"y_shape: {y.shape}")

    n = 1
    W = tf.Variable(tf.random.uniform(shape=(2, n), minval=-1, maxval=1))
    # W = tf.Variable(tf.ones(shape=(2, n)))
    b = tf.Variable(tf.zeros(shape=(n,)))

    print(f"W:\t{W},\nb:\t{b}\n")

    y_pred = dense(W, b, x)
    print(f"y_pred: {y_pred}")


if __name__ == '__main__':
    main()
