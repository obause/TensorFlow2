from typing import Tuple
import numpy as np


def get_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """ OR Dataset """
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([[0], [1], [1], [1]])
    return x, y


def to_categorical(y: np.ndarray, num_classes: int) -> np.ndarray:
    """ to categorical """
    y_cat = np.zeros((y.shape[0], num_classes))
    # y_cat[np.arange(y.shape[0]), y] = 1
    # Andere Option:
    for i, yi in enumerate(y):
        y_cat[i, yi] = 1
    return y_cat


def softmax(y_pred: np.ndarray) -> np.ndarray:
    """ softmax """
    probabilities = np.zeros_like(y_pred)
    for i in range(y_pred.shape[0]):
        exps = np.exp(y_pred[i])
        probabilities[i] = exps / np.sum(exps)
    # return np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
    return probabilities


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ cross entropy """
    num_samples = y_true.shape[0]
    loss = float(-np.sum(y_true * np.log(y_pred)) / num_samples)
    return loss


if __name__ == '__main__':
    x, y = get_dataset()
    y_true = to_categorical(y, num_classes=2)

    y_logits = np.array([[10.8, -3.3], [12.2, 21.8], [1.1, 4.9], [7.05, 3.95]])

    y_prob = softmax(y_logits)
    print(f"y_prob: {y_prob}")
    print(f"y_prob.shape: {y_prob.shape}")

    loss = cross_entropy(y_true, y_prob)
    print(f"loss: {loss}")
