import time
from functools import wraps
from pathlib import Path
from typing import Callable, Any

import cv2
from numpy.typing import NDArray


def load_grayscale_image(filepath: Path) -> NDArray:
    return cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)


def show_image(image: NDArray) -> None:
    cv2.imshow("Display", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def timeit(function: Callable) -> Callable:
    @wraps(function)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = function(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {function.__name__} Took {total_time:.4f} seconds")
        return result

    return wrapper
