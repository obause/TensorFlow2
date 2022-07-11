from tensorflow.keras.datasets import mnist

from tf_utils.plotting import display_digit


def main():
    """ Main function. """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    for i in range(3):
        display_digit(X_train[i], label=y_train[i])


if __name__ == '__main__':
    main()
