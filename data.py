import os
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple

class Emotions(Dataset):
    def __init__(self, cvs_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform=transform
        data_file=os.path.join(root_dir, cvs_file)
        with open(data_file, 'r', newline='') as file:
            self._samples = [
                (
                torch.tensor([int(idx) for idx in row [' pixels'].split()], 
                                          dtype=torch.uint8).reshape(48, 48), 
                                          int(row['emotion']) if 'emotion' in row else None,
                                          )
                                          for row in csv.DictReader(file)
                                          ]
    
    def __len__(self):
        return len(self._samples)
    
    def __getitem__(self, idx: int):
        image_tensor, target = self._samples[idx]
        image = Image.fromarray(image_tensor.numpy())

        if self.transform:
            image=self.transform(image)
        return image, target


