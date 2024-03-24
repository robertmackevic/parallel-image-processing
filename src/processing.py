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
        filepath: Path,
        output_dir: Path,
        transform: Callable[[NDArray], NDArray]
) -> None:
    image = transform(load_image(filepath))
    save_image(filepath=output_dir / filepath.name, image=image)


def _split_data_into_chunks(data: List[Any], num_of_chunks: int) -> List[List[Any]]:
    k, m = divmod(len(data), num_of_chunks)
    return [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_of_chunks)]


def _execute(executors: List[Process | Thread]) -> None:
    for executor in executors:
        executor.start()

    for executor in executors:
        executor.join()


def _load_data_conveyor(filepaths: List[Path], loaded_data_queue: Queue) -> None:
    for filepath in filepaths:
        loaded_data_queue.put({
            "filename": filepath.name,
            "image": load_image(filepath)
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


def _save_data_conveyor(output_dir: Path, processed_data_queue: Queue) -> None:
    while True:
        try:
            data = processed_data_queue.get()
        except Empty:
            continue

        if data is None:
            break

        save_image(filepath=output_dir / data["filename"], image=data["image"])


def _run_image_processing_conveyor(
        filepaths: List[Path],
        output_dir: Path,
        transform: Callable[[NDArray], NDArray]
) -> None:
    loaded_data_queue, processed_data_queue = Queue(), Queue()

    _execute([
        Thread(target=_load_data_conveyor, args=(filepaths, loaded_data_queue)),
        Thread(target=_process_data_conveyor, args=(loaded_data_queue, processed_data_queue, transform)),
        Thread(target=_save_data_conveyor, args=(output_dir, processed_data_queue)),
    ])


@timeit
def process_images_sequential(transform: Callable[[NDArray], NDArray]) -> None:
    output_dir = OUTPUT_DIR / transform.__name__
    makedirs(output_dir, exist_ok=True)

    for filename in listdir(DATASET_DIR):
        _load_transform_and_save(DATASET_DIR / filename, output_dir, transform)


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
def process_images_parallel_conveyors(
        transform: Callable[[NDArray], NDArray],
        num_workers: int = cpu_count(),
) -> None:
    output_dir = OUTPUT_DIR / transform.__name__
    makedirs(output_dir, exist_ok=True)

    data_chunks = _split_data_into_chunks(
        data=[DATASET_DIR / filename for filename in listdir(DATASET_DIR)],
        num_of_chunks=num_workers
    )

    _execute([
        Process(target=_run_image_processing_conveyor, args=(data_chunks[i], output_dir, transform))
        for i in range(num_workers)
    ])
