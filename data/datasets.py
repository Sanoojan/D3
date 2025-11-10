import os
import re
import pandas as pd
import albumentations as A
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


def crop_center_by_percentage(image, percentage):
    h, w = image.shape[:2]
    if w > h:
        p = int(w * percentage)
        return image[:, p:w-p]
    else:
        p = int(h * percentage)
        return image[p:h-p, :]


def get_number_from_filename(filename):
    match = re.match(r'(\d+)', filename)
    return int(match.group(1)) if match else float('inf')


def read_video(folder_path, trans):
    """Reads frames from a video folder and applies the SAME transformation to all frames."""
    image_paths = sorted(os.listdir(folder_path), key=get_number_from_filename)
    total_frames = len(image_paths)
    if total_frames < 8:
        raise ValueError(f"Not enough frames in {folder_path}. Found {total_frames}")

    set_frame = 8 if total_frames < 16 else 16
    image_paths = image_paths[:set_frame]

    # Read all frames first
    images = [cv2.imread(os.path.join(folder_path, img)) for img in image_paths]

    # --- Apply SAME random parameters for all frames ---
    replay = trans(image=images[0])
    params = replay['replay']  # get random params from first frame
    frames = []
    for img in images:
        augmented = A.ReplayCompose.replay(params, image=img)
        image = augmented["image"]
        frames.append(image.transpose(2, 0, 1)[np.newaxis, :])

    frames = np.concatenate(frames, 0)
    frames = torch.tensor(frames[np.newaxis, :]).squeeze(0)
    return frames


def set_preprocessing(aug_type=None, aug_quality=None):
    """Define realistic augmentations for deepfake evaluation."""
    # aug_list = [A.Resize(384, 384)]
    aug_list= [A.Resize(224, 224)]

    # Controlled degradations (optional)
    if aug_type == 'Gaussian_blur':
        aug_list.append(A.GaussianBlur(blur_limit=(3, 7), p=1.0))
    elif aug_type == 'JPEG_compression':
        aug_list.append(A.ImageCompression(quality_lower=aug_quality, quality_upper=aug_quality, p=1.0))
    elif aug_type == 'custom_series':
        # Default evaluation-time realistic augmentations
        aug_list.extend([
            A.OneOf([
                A.MotionBlur(p=0.3),
                A.MedianBlur(blur_limit=5, p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.4)
            ], p=0.4),
            A.OneOf([
                A.ImageCompression(quality_lower=40, quality_upper=80, p=0.4),
                A.Downscale(scale_min=0.7, scale_max=0.9, p=0.4)
            ], p=0.4),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.ISONoise(p=0.2),
        ])

    # Normalization
    aug_list.append(A.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225),
                                max_pixel_value=255.0))

    return A.ReplayCompose(aug_list)


class D3_dataset_AP(Dataset):
    def __init__(self, real_csv, fake_csv, max_len=9999999, aug_type=None, aug_quality=None):
        super(D3_dataset_AP, self).__init__()
        df_real = pd.read_csv(real_csv).head(max_len)
        df_fake = pd.read_csv(fake_csv).head(max_len)
        self.df = pd.concat([df_real, df_fake], axis=0, ignore_index=True)
        self.trans = set_preprocessing(aug_type, aug_quality)
        self.num_fake_samples = df_fake.shape[0]
        self.num_real_samples = df_real.shape[0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        label = row['label']
        frame_path = row['content_path']
        frames = read_video(frame_path, self.trans)
        return frames, label