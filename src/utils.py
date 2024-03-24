import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps, partial
from os import remove
from pathlib import Path
from typing import Callable, Any, List, Type, Optional
from zipfile import ZipFile

import cv2
import pandas as pd
import requests
import seaborn as sns
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
        function(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {function.__name__} "
              f"with args {[v.__name__ for v in kwargs.values() if hasattr(v, '__name__')]} "
              f"took {total_time:.4f} seconds")
        return total_time

    return wrapper


def _generate_worker_sequence(max_workers: int) -> List[int]:
    sequence, workers = [], 2
    while workers <= max_workers:
        sequence.append(workers)
        workers *= 2
    return sequence


def run_testing(
        function: Callable,
        transforms: List[Callable[[NDArray], NDArray]],
        max_workers: int,
        pool_executor: Optional[Type[ProcessPoolExecutor | ThreadPoolExecutor]] = None
) -> None:
    worker_sequence = _generate_worker_sequence(max_workers)
    _function = partial(function, pool_executor=pool_executor) if pool_executor is not None else function

    processing_times = {
        transform.__name__: [
            _function(transform=transform, num_workers=workers)
            for workers in worker_sequence
        ]
        for transform in transforms
    }

    df = pd.DataFrame(processing_times)
    df["Number of Workers"] = worker_sequence
    df = pd.melt(df, id_vars="Number of Workers", var_name="Transform", value_name="Time (seconds)")

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x="Number of Workers", y="Time (seconds)", hue="Transform")

    for patch in ax.patches:
        if patch.get_height() != 0:
            ax.annotate(
                format(patch.get_height(), ".2f"),
                (patch.get_x() + patch.get_width() / 2., patch.get_height()),
                ha="center", va="center", xytext=(0, 10), textcoords="offset points"
            )

    plt.xlabel("Number of Workers", fontsize=14)
    plt.ylabel("Time (seconds)", fontsize=14)

    title = f"Performance of {function.__name__}"
    if pool_executor is not None:
        title += f" using {pool_executor.__name__}"

    plt.title(title, fontsize=16)

    plt.tight_layout()
    plt.show()
