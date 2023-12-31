import os
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
import gdown
import zipfile
import pickle
from PIL import Image

class DIOR(Dataset):

    links = {
        "train": [
            "https://drive.google.com/file/d/1--NeRTtWINde8GURrstElpL0OtLhR80J/view?usp=sharing",
            "train.zip",
            "DIOR"
        ],
        "test": [
            "https://drive.google.com/file/d/1-3J5vJvzn24Aj2thEQm_qyrMv1PjDas4/view?usp=sharing",
            "test.zip",
            "DIOR"
        ]
    }

    def __init__(self, root, train=True, download=False, transform=None, target_transform=None, count=-1, verbose=False, **kwargs):
        super(DIOR, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.verbose = verbose
        self.count = count
        if download:
            self._download_and_extract()
        self.data, self.targets, self.classes = self._load_data()
        self._balance_data()
    
    def _download_and_extract(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        if self.train:
            link = self.links["train"]
        else:
            link = self.links["test"]

        file_path = os.path.join(self.root, link[1])

        if not os.path.exists(file_path):
            gdown.download(link[0], file_path, quiet=not self.verbose, fuzzy=True)
        
        data_path = os.path.join(self.root, link[2])
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
    
    def _load_data(self):
        if self.train:
            data_path = os.path.join(self.root, self.links["train"][2])
        else:
            data_path = os.path.join(self.root, self.links["test"][2])
        with open(os.path.join(data_path, 'class_names.pkl'), 'rb') as f:
            classes = pickle.load(f)
        targets_name = f"{('train' if self.train else 'test')}_targets.pkl"
        with open(os.path.join(data_path, targets_name), 'rb') as f:
            targets = pickle.load(f)
        img_dir = os.path.join(data_path, 'train' if self.train else 'test')
        data = [os.path.join(img_dir, f'{idx:0>5}.jpg') for idx in range(len(targets))]
        return data, targets, classes
    
    def _balance_data(self):
        if self.count == -1:
            return
        # TODO: Implement data balancing
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_name = self.data[index]
        target = self.targets[index]
        img = Image.open(image_name)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


class DIOR_OOD(DIOR):

    def __init__(self, root, train=True, download=False, transform=None, target_transform=None, count=-1, verbose=False, normal_classes=0, **kwargs):
        super().__init__(root, train, download, transform, target_transform, count, verbose, **kwargs)
        if not isinstance(normal_classes, list):
            normal_classes = [normal_classes]
        self.normal_classes = normal_classes
        if self.verbose:
            print(f'Normal classes: {[self.classes[i] for i in normal_classes]}')
    
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if target in self.normal_classes:
            target = 0
        else:
            target = 1
        return img, target


if __name__ == '__main__':
    from torchvision import transforms
    transform = transforms.Resize([224, 224])
    dior_ood = DIOR_OOD(root='.', train=False, download=False, normal_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17], verbose=True, transform=transform)

    # Visualize some data points
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2

    fig, axs = plt.subplots(4, 4, figsize=(16, 16))
    rand_idx = np.random.choice(range(len(dior_ood)), size=16, replace=False)
    samples = [dior_ood[idx] for idx in rand_idx]
    for i in range(4):
        for j in range(4):
            image, target = samples[4*i+j]
            axs[i, j].imshow(image)
            axs[i, j].title.set_text(target)
    plt.show()