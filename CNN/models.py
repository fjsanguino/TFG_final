import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
'''
class Stractor(nn.Module):

    def __init__(self):
        super(Stractor, self).__init__()

        #declare layers used in this network
        # first layer: resnet18 gets a pretrained model
        self.resnet50 = models.resnet50(pretrained=True)
        #        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, 2048)
        # or instead of the above function you can use:
        # self.resnet18.fc = Identity()
        # self.resnet18.avgPool = Identity()



    def forward(self, img):
        x = self.resnet50(img)


        return x
'''

class Stractor(nn.Module):

    def __init__(self):
        super(Stractor, self).__init__()

        ''' declare layers used in this network'''''
        # first block
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=3)#, padding=1)  # 64x64 -> 64x64
        self.bn1 = nn.BatchNorm2d(96)
        #self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 -> 32x32

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1)#, padding=1) # 32x32 -> 32x32
        self.bn2 = nn.BatchNorm2d(256)
        #self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1)#, padding=1) #
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1)#, padding=1) #
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1)#, padding=1) #
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) #

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 11)

    def forward(self, img):
        #print('--------------------------------------')
        #img = img.view(1, img.size(0), img.size(1), img.size(1))
        #print(img.shape)


        x = self.conv1(img)
        #print('After nn.Conv2d(in_channel=3, out_channel=96, kernel_size=3, stride=1, padding=1)', x.shape)
        x = self.bn1(x)
        #print('After nn.BatchNorm2d(96)', x.shape)
        #x = self.relu1(x)
        #print('After nn.ReLU()', x.shape)
        x = self.maxpool1(x)
        #print('After nn.MaxPool2d(kernel_size=2, stride=2)', x.shape)

        x = self.conv2(x)
        #print('After nn.Conv2d(in_channel=48, out_channel=256, kernel_size=3, stride=1, padding=1)', x.shape)
        x = self.bn2(x)
        #print('After nn.BatchNorm2d(256)', x.shape)
        #x = self.relu2(x)
        #print('After nn.ReLU()', x.shape)
        x = self.maxpool2(x)
        #print('After nn.MaxPool2d(kernel_size=2, stride=2)', x.shape)

        x = self.conv3(x)
        #print('After nn.Conv2d(128, 384, kernel_size=3, stride=1)', x.shape)
        x = self.conv4(x)
        #print('After nn.Conv2d(384, 384, kernel_size=3, stride=1)', x.shape)
        x = self.conv5(x)
        #print('After nn.Conv2d(384, 256, kernel_size=3, stride=1)', x.shape)

        x = self.maxpool3(x)
        #print('After nn.MaxPool2d(kernel_size=2, stride=2)',x.shape)
        x = x.view(x.size(0), -1)
        #print('After view', x.shape)
        x = self.fc1(x)
        #print('After nn.Linear(1024, 1024)', x.shape)
        #x = self.fc2(x)
        #print('After nn.Linear(1024, 11)', x.shape)

        return x
class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(2048, 2048)
        #self.norm1 = nn.BatchNorm1d(1024)
        #self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2048, 15)
        self.norm2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 256)
        self.norm3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(256, 128)
        self.norm4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(128, 64)
        self.norm5 = nn.BatchNorm1d(64)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(64, 15)

    def forward(self, feat):
        #x = self.relu1(self.norm1(self.fc1(feat)))
        #print(feat.shape)
        x = self.fc1(feat)
        output = self.fc2(x)
        #x = self.relu2(self.norm2(self.fc2(feat)))
        #print(x.shape)
        #x = self.relu3(self.norm3(self.fc3(x)))
        #print(x.shape)
        #x = self.relu4(self.norm4(self.fc4(x)))
        #print(x.shape)
        #x = self.norm5(self.fc5(x))
        #output = self.fc6(x)
        #print(output.shape)


        return x, output