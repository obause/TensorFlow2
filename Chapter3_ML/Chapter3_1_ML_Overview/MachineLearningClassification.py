import matplotlib.pyplot as plt
import numpy as np

from tf_utils.dummyData import classification_data


def model(x: np.ndarray) -> np.ndarray:
    m = -6.0  # slope
    b = 12.0  # intercept
    return m * x + b


if __name__ == '__main__':
    x, y = classification_data()

    y_pred = model(x)

    colors = np.array(['r', 'b'])

    plt.scatter(x[:, 0], x[:, 1], c=colors[y])
    plt.plot(x, y_pred)
    plt.show()
