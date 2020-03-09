import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

import data
import models
import parser
from test import evaluate

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
                fea = torch.cat((fea[0], fea[1]), 0)
                #print(fea.shape)
                features.append(fea)
            features = torch.stack(features)
            #print(features.shape)

            _, output = classifier(features)
            print(output.shape)
            #print(output.shape)

            label = label.cuda()

            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            print(train_info)
            f = open(os.path.join(args.save_dir, 'log.txt'), "a+")
            f.write(train_info + '\n')


    if epoch % args.val_epoch == 0:
        acc = evaluate(classifier, val_loader)
        val_info = 'Epoch: [{}] ACC:{}'.format(epoch, acc)
        print(val_info)
        f = open(os.path.join(args.save_dir, 'log.txt'), "a+")
        f.write(val_info + '\n')

        if acc > best_acc:
            save_model(classifier, os.path.join(args.save_dir, 'model_best_class.pth.tar'))
            save_model(feature_stractor, os.path.join(args.save_dir, 'model_best_stractor.pth.tar'))

            best_acc = acc

    save_model(classifier, os.path.join(args.save_dir, 'model_{}_class.pth.tar'.format(epoch)))
    save_model(feature_stractor, os.path.join(args.save_dir, 'model_{}_stractor.pth.tar'.format(epoch)))




