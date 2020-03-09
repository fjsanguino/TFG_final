import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


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


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=2048, h_RNN_layers=1, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=11):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, sequence, n_frames):
        packed = torch.nn.utils.rnn.pack_padded_sequence(sequence, n_frames, batch_first=True)
        RNN_out, (h_n, h_c) = self.LSTM(packed, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """
        #print(RNN_out[-1].shape)
        # FC layers
        x = self.fc1(h_n[-1])  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        output = self.fc2(x)

        return x, output



