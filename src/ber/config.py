"""Paths and global settings for Business Entity Resolution pipeline."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get('BER_DATA', ROOT / 'student_resource' / 'dataset'))
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR = DATA_DIR / 'test'
WORK_DIR = Path(os.environ.get('BER_WORK', ROOT / 'work'))
OUT_DIR = Path(os.environ.get('BER_OUT', ROOT / 'output'))

# Conservative 4 parallel workers to guarantee zero paging/memory exhaustion on Windows
N_JOBS = int(os.environ.get('BER_JOBS', 4))
SEED = 42
