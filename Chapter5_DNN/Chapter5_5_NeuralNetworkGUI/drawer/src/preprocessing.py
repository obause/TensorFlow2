import os
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import center_of_mass


FILE_PATH = os.path.abspath(__file__)
PROJECT_DIR = os.path.dirname(os.path.dirname(FILE_PATH))


def load(image_path: str) -> np.ndarray:
    """ Load an image from a path """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


def resize(image: np.ndarray) -> np.ndarray:
    """ Resize an image to a 28x28 image """
    image = cv2.resize(image, (28, 28))
    return image


def normalize(image: np.ndarray) -> np.ndarray:
    """Invert the image"""
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    # Alternativ: image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
    return image


def center(image: np.ndarray) -> np.ndarray:
    cy, cx = center_of_mass(image) # x, y Kooridinaten von der Mitte der Zahl
    rows, cols = image.shape
    # Verschiebung der x und y Koordinaten berechnen
    shiftx = np.round(cols / 2 - cx).astype(int)
    shifty = np.round(rows / 2 - cy).astype(int)
    M = np.array([[1, 0, shiftx], [0, 1, shifty]]).astype(np.float32) # Translationsmatrix
    image = cv2.warpAffine(image, M, (cols, rows))
    return image


def get_image(DrawingFrame: Any, debug: bool = False) -> np.ndarray:
    pixmap = DrawingFrame.grab()
    temp_image_file_path = os.path.join(PROJECT_DIR, "ressources", "imgs", "temp_image.jpg")
    pixmap.save(temp_image_file_path)
    image = load(temp_image_file_path)
    image = resize(image)
    image = normalize(image)
    image = center(image)
    plt.imshow(image, cmap="gray")
    plt.show()
    return image
