import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import pandas as pd

# Example Dataset class that provides (photo, xray, label) pairs
class DentalDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.data = []

        for label, category in enumerate(['Extraction', 'Non_Extraction']):
            category_path = os.path.join(root, category)
            xray_paths = sorted(os.listdir(os.path.join(category_path, 'Xray')))
            photo_l_paths = sorted(os.listdir(os.path.join(category_path, 'Photo_L')))
            photo_u_paths = sorted(os.listdir(os.path.join(category_path, 'Photo_U')))
            metadata_paths = sorted(os.listdir(os.path.join(category_path, 'MetaData')))

            for xray, photo_l, photo_u, meta in zip(xray_paths, photo_l_paths, photo_u_paths, metadata_paths):
                xray_image = self.load_image(os.path.join(category_path, 'Xray', xray), channels=1)
                photo_l_image = self.load_image(os.path.join(category_path, 'Photo_L', photo_l), channels=3)
                photo_u_image = self.load_image(os.path.join(category_path, 'Photo_U', photo_u), channels=3)
                metadata = pd.read_excel(os.path.join(category_path, 'MetaData', meta))
                xray_image = transform.functonal.center_crop(xray_image, (867, 590))
                photo_l_image = transform.functonal.center_crop(photo_l_image, (887, 1355))
                photo_u_image = transform.functonal.center_crop(photo_u_image, (887, 1355))

                if self.transform:
                    xray_image = self.transform(xray_image)
                    photo_l_image = self.transform(photo_l_image)
                    photo_u_image = self.transform(photo_u_image)

                self.data.append({
                    'xray': xray_image,
                    'photo_l': photo_l_image,
                    'photo_u': photo_u_image,
                    'meta': metadata,
                    'label': label
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        xray = item['xray']
        photo_l = item['photo_l']
        photo_u = item['photo_u']
        metadata = item['meta']
        label = torch.tensor(item['label'], dtype=torch.float32)

        return xray, photo_l, photo_u, metadata, label

    def load_image(self, path, channels):
        image = Image.open(path).convert('RGB' if channels == 3 else 'L')
        return image