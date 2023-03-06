import json
from time import time
import multiprocessing as mp

import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from torch.utils.data import DataLoader
from functools import lru_cache

from models.utils.pooling import convert_points_2_two, calculate_rectangular_coordinates
from utils_funcs import utils


class ACPDS():
    """
    A basic dataset of parking lot images,
    parking space coordinates (ROIs), and occupancies.
    Returns the tuple (image, rois, occupancy).
    """
    def __init__(self, dataset_path, ds_type='train', res=None):
        self.dataset_path = dataset_path
        self.ds_type = ds_type
        self.res = res

        # load all annotations
        with open(f'{self.dataset_path}/annotations.json', 'r') as f:
            all_annotations = json.load(f)

        # select train, valid, test, or all annotations
        if ds_type in ['train', 'valid', 'test']:
            # select train, valid, or test annotations
            annotations = all_annotations[ds_type]
        else:
            # select all annotations
            assert ds_type == 'all'
            # if using all annotations, combine the train, valid, and test dicts
            annotations = {k:[] for k in all_annotations['train'].keys()}
            for ds_type in ['train', 'valid', 'test']:
                for k, v in all_annotations[ds_type].items():
                    annotations[k] += v

        self.fname_list = annotations['file_names']
        self.rois_list = annotations['rois_list']
        self.occupancy_list = annotations['occupancy_list']

    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        # load image
        image_path = f'{self.dataset_path}/images/{self.fname_list[idx]}'
        image = torchvision.io.read_image(image_path)
        if self.res is not None:
            image = TF.resize(image, self.res)
            
        # load occupancy
        occupancy = self.occupancy_list[idx]
        occupancy = torch.tensor(occupancy, dtype=torch.int64)

        # load rois
        rois = self.rois_list[idx]
        rois = torch.tensor(rois)
        C, H, W = image.shape

        rois[..., 0] *= (W - 1)
        rois[..., 1] *= (H - 1)

        # Project quadrilaterals to minimum rectangle
        rois = [calculate_rectangular_coordinates(roi[0], roi[1], roi[2], roi[3]) for roi in rois]

        rois = convert_points_2_two(rois)

        # getting the areas of the boxes
        area = (rois[:, 3] - rois[:, 1]) * (rois[:, 2] - rois[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((rois.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = rois
        target["labels"] = occupancy
        target["area"] = area
        target["iscrowd"] = iscrowd
        # image_id
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
    
        return image, target
    
    def __len__(self):
        return len(self.fname_list)


def collate_fn(batch):
    images = [item[0] for item in batch]
    rois = [item[1] for item in batch]
    occupancy = [item[2] for item in batch]
    return [images, rois, occupancy]


def create_datasets(dataset_path, batch_size, *args, **kwargs):
    """
    Create training and test DataLoaders.
    Returns the tuple (image, rois, occupancy).
    During the first pass, the DataLoaders will be cached.
    """
    ds_train = ACPDS(dataset_path, 'train', *args, **kwargs)
    ds_valid = ACPDS(dataset_path, 'valid', *args, **kwargs)
    ds_test = ACPDS(dataset_path, 'test', *args, **kwargs)

    data_loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=utils.collate_fn)
    data_loader_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False, collate_fn=utils.collate_fn)
    data_loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, collate_fn=utils.collate_fn)
    return data_loader_train, data_loader_valid, data_loader_test

def get_all_possible_num_of_workers(ds):
    for num_workers in range(2, mp.cpu_count(), 2):
        train_loader = DataLoader(ds,shuffle=True,num_workers=num_workers,batch_size=64,pin_memory=True)
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))