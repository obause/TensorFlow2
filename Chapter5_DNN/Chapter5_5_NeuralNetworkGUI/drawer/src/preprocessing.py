import os
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import center_of_mass


FILE_PATH = os.path.abspath(__file__)
PROJECT_DIR = os.path.dirname(os.path.dirname(FILE_PATH))


def load(image_path: str) -> np.ndarray:
    pass  # TODO


def resize(image: np.ndarray) -> np.ndarray:
    pass  # TODO


def normalize(image: np.ndarray) -> np.ndarray:
    pass  # TODO


def center(image: np.ndarray) -> np.ndarray:
    pass  # TODO


def get_image(DrawingFrame: Any, debug: bool = False) -> np.ndarray:
    pixmap = DrawingFrame.grab()
    temp_image_file_path = os.path.join(PROJECT_DIR, "ressources", "img", "temp.png")
    pixmap.save(temp_image_file_path)
    image = load(temp_image_file_path)
    image = resize(image)
    image = normalize(image)
    image = center(image)
    return image
