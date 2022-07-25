import enum
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import MaxPool2D


def max_pooling(image: np.ndarray) -> np.ndarray:
    rows, cols = image.shape
    print(f"rows: {rows}, cols: {cols}")
    output = np.zeros(shape=(rows // 2, cols // 2), dtype=np.float32) # Outputdimension
    for i_out, i in enumerate(range(0, rows, 2)):
        for j_out, j in enumerate(range(0, cols, 2)):
            output[i_out, j_out] = np.max(image[i: i + 2, j: j + 2])
    return output

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image = x_train[0]
    image = image.reshape((28, 28)).astype(np.float32)

    pooling_image = max_pooling(image)

    print(f"Prvious shape: {image.shape} current shape: {pooling_image.shape}")
    print(f"Pooled Image:\n{pooling_image.squeeze()}")

    layer = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')
    pooling_image_tf = layer(image.reshape((1, 28, 28, 1))).numpy()
    print(f"Pooled Image TF:\n{pooling_image_tf.squeeze()}")
    # assert np.allclose(pooling_image.flatten(), pooling_image_tf.flatten())

    fig, axs = plt.subplots(nrows=1, ncols=3)
    axs[0].imshow(image, cmap="gray")
    axs[1].imshow(pooling_image, cmap="gray")
    axs[2].imshow(pooling_image_tf.squeeze(), cmap="gray")
    plt.show()
