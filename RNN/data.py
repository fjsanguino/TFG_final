import os
import json
import torch
import scipy.misc

import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

from utils import getVideoList, readShortVideo

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


class DATA(Dataset):
    def __init__(self, args, split='split0'):

        ''' set up basic parameters for dataset '''
        self.split = split
        self.data_dir = args.data_dir
        self.video_dir = os.path.join(self.data_dir, self.split)
        #print(self.video_dir)

        ''' read the data list '''
        self.label_path = os.path.join(self.video_dir, 'labels_' + split + '.csv')

        self.dic = getVideoList(self.label_path)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD)
        ])


    def __len__(self):
        return len(self.dic.get('video_name'))

    def __getitem__(self, idx):

        video = {}
        for x, y in self.dic.items():
            video[x] = y[idx]

        return video

csv_file = '/media/fj-sanguino/Elements/Breakfast/Splits/split0/labels_split0.csv'


