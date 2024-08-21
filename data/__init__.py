import datasets
import numpy as np
import os
import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class CustomDataset(Dataset):
    def __init__(self, features_dir, captions_dir, masks_dir):
        self.features_dir = features_dir
        self.captions_dir = captions_dir
        self.masks_dir = masks_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.captions_files = sorted(os.listdir(captions_dir))
        self.masks_files = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]

        label_idx = int(os.path.splitext(feature_file)[0].split("-")[0])
        caption_file = np.random.choice(list(filter(lambda x: x.startswith(f"{label_idx}-"), self.captions_files)))
        mask_file = caption_file

        features = np.load(os.path.join(self.features_dir, feature_file))
        captions = np.load(os.path.join(self.captions_dir, caption_file))
        masks = np.load(os.path.join(self.masks_dir, mask_file))
        return torch.from_numpy(features), torch.from_numpy(captions), torch.from_numpy(masks)
