import os
import json
import torch
import scipy.misc
import numpy as np

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
        self.label_path = os.path.join(self.video_dir, 'labels_' + split + '_new.csv')

        self.dic = getVideoList(self.label_path)

        self.transform = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            transforms.Resize(170),
            transforms.CenterCrop(170),
            transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD)
        ])


    def __len__(self):
        return len(self.dic.get('video_name'))

    def __getitem__(self, idx):

        video = {}
        for x, y in self.dic.items():
            video[x] = y[idx]

        frames = readShortVideo(self.data_dir, video.get('clip_name'))
        #print(video.get('clip_name'), frames.shape[0])
        random1 = np.random.randint(low=0, high=frames.shape[0])
        random2 = np.random.randint(low=0, high=frames.shape[0])

        #print(frames.shape, random1, random2)

        if frames.shape[0] > 1:
            while random1 == random2:
                #print('here')
                random2 = np.random.randint(low=0, high=frames.shape[0])

        frames = [frames[random1], frames[random2]]
        frames = np.asarray(frames)
        vid = []
        for frame in frames:
            frame = self.transform(frame)
            vid.append(frame)
        vid = torch.stack(vid)
        #print(vid.shape)

        label = video.get('label_number')
        #print(vid.type())
        return vid, int(label)




