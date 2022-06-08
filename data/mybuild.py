import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
from randaugment import RandAugment
from PIL import Image
from timm.data import Mixup

from .samplers import SubsetRandomSampler

class OrchidDataset(Dataset):
    """Orchid dataset."""

    def __init__(self, root_dir, landmarks, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = landmarks
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name, landmark = self.landmarks_frame.iloc[idx, :]
        img_path = os.path.join(self.root_dir, img_name)
        # image = io.imread(img_path)
        image = Image.open(img_path)
        # convert to PIL
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # im_pil = Image.fromarray(img)
        # sample = {'image': image, 'landmarks': landmark}

        if self.transform:
            image = self.transform(image)

        return image, landmark

def read_dataset(config, ratio=0.9, seed=0):
    data_dir = './datasets/' + config.DATA.DATASET
    
    transform = {
        'train': transforms.Compose([
            transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
            # CutoutPIL(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
            transforms.ToTensor(),
        ])}

    landmarks_frame = pd.read_csv(data_dir+'/label.csv')
    train_landmarks, val_landmarks = train_test_split(landmarks_frame, train_size = int(len(landmarks_frame) * ratio), shuffle = True, random_state=seed)

    dataset_train = OrchidDataset(data_dir,
                        train_landmarks,
                        transform['train'])
    dataset_val = OrchidDataset(data_dir,
                        val_landmarks,
                        transform['val'])

    print("len(dataset_train)): ", len(dataset_train))
    print("len(dataset_val)): ", len(dataset_val))

    # Pytorch Data loader
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    
    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)
    
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        # shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        # batch_size=1,
        batch_size=config.DATA.BATCH_SIZE,
        # shuffle=False,
    )
    
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, train_loader, val_loader, mixup_fn