import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

from tf_utils.dummyData import regression_data


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean((y_true - y_pred)**2)


if __name__ == '__main__':
    x, y = regression_data()
    x = x.reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    reg = LinearRegression().fit(x_train, y_train)
    score = reg.score(x_test, y_test)
    print(f"Score: {score}")
    print(f"Coef: {reg.coef_}")
    print(f"Intercept: {reg.intercept_}")

    y_pred = reg.predict(x_test)

    mae_score = mae(x_test, y_pred)
    mse_score = mse(x_test, y_pred)

    print(f"MAE: {mae_score}")
    print(f"MSE: {mse_score}")

    plt.scatter(x, y, c='r')
    plt.plot(x_test, y_pred, c='b')
    plt.show()
