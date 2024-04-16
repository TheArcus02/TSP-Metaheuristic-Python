from typing import Tuple

import numpy as np


def calculate_distance(point1: Tuple[int, int], point2: Tuple[int, int]):
    """
    Calculate the distance between two points
    :param point1:
    :param point2:
    :return:
    """
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
