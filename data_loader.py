import os
import glob
from typing import List, Tuple
from torchvision import transforms

TARGET_SIZE = (128,128)

def load_imagefolder_paths(root_dir: str) -> Tuple[List[str], List[str]]:
    """
    Return list of files and corresponding labels from an ImageFolder-style root.
    It searches recursively for files under root_dir/*/*.*
    """
    exts = ("*.png","*.jpg","*.jpeg","*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(root_dir, "*", e)))
    files = sorted(files)
    labels = [os.path.basename(os.path.dirname(p)) for p in files]
    return files, labels

def make_dataloaders(root_dir: str, batch_size: int = 16, val_split: float = 0.2):
    # kept minimal, training function uses ImageFolder directly
    raise NotImplementedError("Use models.trainer.train_loop which handles dataloaders from ImageFolder.")
