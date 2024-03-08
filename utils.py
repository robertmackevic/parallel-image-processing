from pathlib import Path

import cv2
from numpy.typing import NDArray


def load_grayscale_image(filepath: Path) -> NDArray:
    return cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)


def show_image(image: NDArray) -> None:
    cv2.imshow("Display", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
