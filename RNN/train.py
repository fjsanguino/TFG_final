import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os

import models
import data
import parser


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

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


args = parser.arg_parse()

'''create directory to save trained model and other info'''
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)


''' setup GPU '''
torch.cuda.set_device(args.gpu)


''' setup random seed '''
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)


''' load dataset and prepare data loader '''
print('===> prepare data ...')
data_loader = data.DATA(args, split='split0')

splits = ['split0', 'split1', 'split2', 'split3']
splits.remove(splits[splits.index(args.val_split)])

train_data = torch.utils.data.ConcatDataset([data.DATA(args, split=splits[0]),
                                             data.DATA(args, split=splits[1]),
                                             data.DATA(args, split=splits[2])])
val_data =  data.DATA(args, split=args.val_split)

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=args.train_batch,
                                           num_workers=4,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(val_data,
                                           batch_size=args.train_batch,
                                           num_workers=4,
                                           shuffle=False)

''' load model '''
print('===> prepare model ...')


feature_stractor = models.Stractor()
feature_stractor = feature_stractor.cuda()
#feature_stractor = feature_stractor.eval()

classifier = models.Classifier()
classifier = classifier.cuda()

''' define loss '''
criterion = nn.CrossEntropyLoss()

''' setup optimizer '''
params = list(classifier.parameters()) + list(feature_stractor.parameters())
optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

iters = 0
accs = []
best_acc = 0
print('===> begin training ...')
for epoch in range(1, args.epoch + 1):
    for idx, (video, label) in enumerate(train_loader): #loads several videos
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(train_loader))

            iters += 1

            video = video.cuda()
            #print(video.shape)
            features = []
            for v in video:
                fea = feature_stractor(v)
                features.append(fea)

