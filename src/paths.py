from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"

TEST_IMAGE_FILEPATH = DATA_DIR / "test_image.jpg"
DATASET_DOWNLOAD_PATH = DATA_DIR / "dataset.zip"
DATASET_DIR = DATA_DIR / "dataset" / "Images"
