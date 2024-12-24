from os import listdir, walk
from os.path import isfile, join

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from PIL import Image
import numpy as np

class ImageDataSet(Dataset):
    def __init__(self, root='test', transform=None, subdir=False):
        self.root = root
        self.transform = transform

        if subdir:
            self.images = self.__subdirectory()
        else:
            self.images = [f for f in listdir(self.root) if isfile(join(self.root, f))]

    def __len__(self):
        return len(self.images)

    def __subdirectory(self):
        dirs = []
        for path, subdirs, files in walk(self.root):
            for name in files:
                dirs.append(join(path, name))

        return dirs

    def __getitem__(self, index):
        # path = join(self.root, self.images[index])
        path = self.images[index]  # for linux
        img = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)

        return img, path