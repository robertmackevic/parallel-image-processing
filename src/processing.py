from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
from os import listdir, makedirs
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from typing import Callable, Type, Optional

from numpy.typing import NDArray

from src.paths import OUTPUT_DIR, DATASET_DIR
from src.utils import timeit, load_image, save_image


def _load_transform_and_save(
        image_filepath: Path,
        transform: Callable[[NDArray], NDArray],
        output_dir: Path
) -> None:
    image = load_image(image_filepath)
    image = transform(image)
    save_image(filepath=output_dir / image_filepath.name, image=image)
    return


def _bulk_load_transform_and_save(
        image_filepaths: Path,
        transform: Callable[[NDArray], NDArray],
        output_dir: Path
) -> None:
    for image_filepath in image_filepaths:
        image = load_image(image_filepath)
        image = transform(image)
        save_image(filepath=output_dir / image_filepath.name, image=image)
    return

def chunk_data(image_paths: list[Path], num_of_chunks: int) -> list[list[Path]]:
    k, m = divmod(len(image_paths), num_of_chunks)
    return (image_paths[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_of_chunks))


@timeit
def process_images_sequential(transform: Callable[[NDArray], NDArray]) -> None:
    output_dir = OUTPUT_DIR / transform.__name__
    makedirs(output_dir, exist_ok=True)

    for filename in listdir(DATASET_DIR):
        _load_transform_and_save(DATASET_DIR / filename, transform, output_dir)


@timeit
def process_images_parallel_1(
        transform: Callable[[NDArray], NDArray],
        pool_executor: Type[ProcessPoolExecutor | ThreadPoolExecutor],
        num_workers: Optional[int] = None
) -> None:
    output_dir = OUTPUT_DIR / transform.__name__
    makedirs(output_dir, exist_ok=True)

    with pool_executor(num_workers) as executor:
        executor.map(
            partial(_load_transform_and_save, transform=transform, output_dir=output_dir),
            [DATASET_DIR / filename for filename in listdir(DATASET_DIR)]
        )


def _load_data(loaded_data_queue: Queue) -> None:
    for filename in listdir(DATASET_DIR):
        loaded_data_queue.put({
            "filename": filename,
            "image": load_image(DATASET_DIR / filename)
        })

    loaded_data_queue.put(None)


def _process_data(
        loaded_data_queue: Queue,
        processed_data_queue: Queue,
        transform: Callable[[NDArray], NDArray],
) -> None:
    while True:
        try:
            data = loaded_data_queue.get()
        except Empty:
            continue

        if data is None:
            break

        processed_data_queue.put({
            "filename": data["filename"],
            "image": transform(data["image"])
        })

    processed_data_queue.put(None)


def _save_data(
        processed_data_queue: Queue,
        output_dir: Path
) -> None:
    while True:
        try:
            data = processed_data_queue.get()
        except Empty:
            continue

        if data is None:
            break

        save_image(filepath=output_dir / data["filename"], image=data["image"])


@timeit
def process_images_parallel_2(
        transform: Callable[[NDArray], NDArray],
) -> None:
    output_dir = OUTPUT_DIR / transform.__name__
    makedirs(output_dir, exist_ok=True)
    loaded_data_queue, processed_data_queue = Queue(), Queue()

    loading_thread = Thread(target=_load_data, args=(loaded_data_queue,))
    loading_thread.start()

    processing_thread = Thread(target=_process_data, args=(loaded_data_queue, processed_data_queue, transform))
    processing_thread.start()

    storing_thread = Thread(target=_save_data, args=(processed_data_queue, output_dir))
    storing_thread.start()

    loading_thread.join()
    processing_thread.join()
    storing_thread.join()


@timeit
def process_images_parallel_no_load_balancing(
        transform: Callable[[NDArray], NDArray],
        pool_executor: Type[ProcessPoolExecutor | ThreadPoolExecutor],
        num_workers: Optional[int] = None
) -> None:
    output_dir = OUTPUT_DIR / transform.__name__
    makedirs(output_dir, exist_ok=True)

    image_paths = [DATASET_DIR / filename for filename in listdir(DATASET_DIR)]
    chunks = chunk_data(image_paths, num_workers if num_workers != None else 8)
    
    with pool_executor(num_workers) as executor:
        for chunk in chunks:
            executor.submit(partial(_bulk_load_transform_and_save, transform=transform, output_dir=output_dir), chunk)

