import torch
import torchvision.transforms as transforms
import torch.nn as nn

import numpy as np
from sklearn.metrics import accuracy_score

from utils import readShortVideo

def transforms_array(array):
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize(170),
        transforms.CenterCrop(170),
        transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
        transforms.Normalize(MEAN, STD)
    ])
    return transform(array)

def batch_padding(batch_fea, batch_cls):
    n_frames = [fea.shape[0] for fea in batch_fea]
    perm_index = np.argsort(n_frames)[::-1]

    # sort by sequence length
    batch_fea_sort = [batch_fea[i] for i in perm_index]
    #print(len(batch_fea_sort))
    n_frames = [fea.shape[0] for fea in batch_fea_sort]
    padded_sequence = nn.utils.rnn.pad_sequence(batch_fea_sort, batch_first=True)
    label = torch.tensor(np.array(batch_cls)[perm_index])
    return padded_sequence, label, n_frames




def evaluate(feature_stractor, rnn, data_loader, data_dir):
    feature_stractor.eval()
    rnn.eval()
    preds = []
    gts = []
    with torch.no_grad():  # do not need to caculate information for gradient during eval
        for idx, (videos) in enumerate(data_loader):  # loads several videos
            label = []
            features = []
            for i in range(len(videos)):
                # print('working in video', i + 1, '/', idx + 1)
                frames = readShortVideo(data_dir, videos.get('clip_name')[i])
                # print(frames.shape)
                vid = []
                for j in range(frames.shape[0]):
                    im = transforms_array(frames[j])
                    vid.append(im)
                vid = torch.stack(vid)
                vid = vid.cuda()
                print('working in video ', videos.get('video_index')[i], ' with size ', vid.shape)
                feature = feature_stractor(vid)
                # print(feature.shape)
                features.append(feature)
                label.append(int(videos.get('label_number')[i]))

            sequence, label, n_frames = batch_padding(features, label)

            sequence = sequence.cuda()

            _, pred = rnn(sequence, n_frames)

            _, pred = torch.max(pred, dim=1)
            #print(pred.shape)
            pred = pred.cpu().numpy().squeeze()
            preds.append(pred)
            gts.append(label)

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)
    #print(preds)
    return accuracy_score(gts, preds)



