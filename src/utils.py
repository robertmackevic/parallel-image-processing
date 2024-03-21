import time
from functools import wraps
from os import remove
from pathlib import Path
from typing import Callable, Any
from zipfile import ZipFile

import cv2
import requests
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tqdm import tqdm

from src.paths import DATASET_DOWNLOAD_PATH, DATASET_DIR


def download_dataset_from_dropbox(url: str) -> None:
    if DATASET_DIR.is_dir():
        print("Dataset already exists")
        return

    response = requests.get(url, stream=True)

    if response.status_code != 200:
        print("Download request failed")
        return

    total_size = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)

    with open(DATASET_DOWNLOAD_PATH, "wb") as file:
        for data in response.iter_content(chunk_size=128):
            progress_bar.update(len(data))
            file.write(data)
        progress_bar.close()

    print("Dataset downloaded. Unzipping...")
    extract_dir = DATASET_DOWNLOAD_PATH.with_suffix("")
    extract_dir.mkdir(exist_ok=True)

    with ZipFile(DATASET_DOWNLOAD_PATH, "r") as file:
        file.extractall(extract_dir)

    remove(DATASET_DOWNLOAD_PATH)
    print("Dataset extracted")


def load_image(filepath: Path) -> NDArray:
    return cv2.imread(str(filepath))


def save_image(filepath: Path, image: NDArray) -> None:
    cv2.imwrite(str(filepath), image)


def show_image(image: NDArray) -> None:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()


def timeit(function: Callable) -> Callable:
    @wraps(function)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = function(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {function.__name__} "
              f"with args {[v.__name__ for k, v in kwargs.items()]} "
              f"took {total_time:.4f} seconds")
        return result

    return wrapper
