from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from multiprocessing import Process
from os import listdir, makedirs, cpu_count
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from typing import Callable, Type, Optional, List, Any

from numpy.typing import NDArray

from src.paths import OUTPUT_DIR, DATASET_DIR
from src.utils import timeit, load_image, save_image


def _load_transform_and_save(
        image_filepath: Path,
        transform: Callable[[NDArray], NDArray],
        output_dir: Path
) -> None:
    image = transform(load_image(image_filepath))
    save_image(filepath=output_dir / image_filepath.name, image=image)


def _bulk_load_transform_and_save(
        image_filepaths: List[Path],
        transform: Callable[[NDArray], NDArray],
        output_dir: Path
) -> None:
    for filepath in image_filepaths:
        _load_transform_and_save(filepath, transform, output_dir)


def _split_data_into_chunks(data: List[Any], num_of_chunks: int) -> List[List[Any]]:
    k, m = divmod(len(data), num_of_chunks)
    return [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_of_chunks)]


def _load_data_conveyor(loaded_data_queue: Queue) -> None:
    for filename in listdir(DATASET_DIR):
        loaded_data_queue.put({
            "filename": filename,
            "image": load_image(DATASET_DIR / filename)
        })

    loaded_data_queue.put(None)


def _process_data_conveyor(
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


def _save_data_conveyor(processed_data_queue: Queue, output_dir: Path) -> None:
    while True:
        try:
            data = processed_data_queue.get()
        except Empty:
            continue

        if data is None:
            break

        save_image(filepath=output_dir / data["filename"], image=data["image"])


@timeit
def process_images_sequential(transform: Callable[[NDArray], NDArray]) -> None:
    output_dir = OUTPUT_DIR / transform.__name__
    makedirs(output_dir, exist_ok=True)

    for filename in listdir(DATASET_DIR):
        _load_transform_and_save(DATASET_DIR / filename, transform, output_dir)


@timeit
def process_images_parallel_pooled(
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


@timeit
def process_images_parallel_conveyor(
        transform: Callable[[NDArray], NDArray],
) -> None:
    output_dir = OUTPUT_DIR / transform.__name__
    makedirs(output_dir, exist_ok=True)
    loaded_data_queue, processed_data_queue = Queue(), Queue()

    loading_thread = Thread(target=_load_data_conveyor, args=(loaded_data_queue,))
    loading_thread.start()

    processing_thread = Thread(target=_process_data_conveyor, args=(loaded_data_queue, processed_data_queue, transform))
    processing_thread.start()

    storing_thread = Thread(target=_save_data_conveyor, args=(processed_data_queue, output_dir))
    storing_thread.start()

    loading_thread.join()
    processing_thread.join()
    storing_thread.join()


@timeit
def process_images_parallel_no_load_balancing(
        transform: Callable[[NDArray], NDArray],
        executor: Type[Process | Thread],
        num_workers: int = cpu_count()
) -> None:
    output_dir = OUTPUT_DIR / transform.__name__
    makedirs(output_dir, exist_ok=True)

    data_chunks = _split_data_into_chunks(
        data=[DATASET_DIR / filename for filename in listdir(DATASET_DIR)],
        num_of_chunks=num_workers
    )

    executors = [
        executor(target=_bulk_load_transform_and_save, args=(data_chunks[i], transform, output_dir))
        for i in range(num_workers)
    ]

    for exe in executors:
        exe.start()

    for exe in executors:
        exe.join()
