import torch
import torchvision.transforms as transforms

import os

from PIL import Image
from torch.utils.data import Dataset

class COCO_val(Dataset):
    def __init__(self):
        self.imglist = os.listdir("data/val2017/")
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        img = Image.open("data/val2017/" + self.imglist[idx]).convert("RGB")
        img = self.transform(img)

        return img