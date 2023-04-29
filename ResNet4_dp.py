"""
Resnet8
"""

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

#define convolution 3x3
#def conv2d_3x3(in_channels, out_channels, stride=1):
#    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
#                     stride=stride, padding=1)


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

#Residual class
class blockBasic(nn.Module):
    expansion = 1
    def __init__(self, in_channels, channels, stride=1, option='A', drop=0.0, num_groups=8):
        super(blockBasic, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, channels)
 #       self.dropout = nn.Dropout(drop)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, channels)
#        self.dropout = nn.Dropout(drop)
        self.shortcut = nn.Sequential()
        if (stride != 1) or (in_channels != channels):
            if option == 'A':
                self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride))
 #               nn.GroupNorm(num_groups, channels))
            else: 
                self.shortcut = LambdaLayer(
                lambda x:F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, channels//4, channels//4), "constant", 0))

    def forward(self, x):
        '''
        Input x: a batch of images (batch size x in_channels x features# x features#)
        Return features
        '''
        out = x
        out = self.conv1(x)
        out = self.gn1(out)
 #       out = self.dropout(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.gn2(out)
#        out = self.dropout(out)
        out += self.shortcut(x)
        out = self.act(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_class): #wait to check num_classes
        super(ResNet, self).__init__()
        ### YOUR CODE HERE
        self.layers = layers
        self.drop = 0
        self.block = block
        self.num_class = num_class
        self.in_channels = 64
        self.conv00 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn00 = nn.GroupNorm(3, 3)
        self.conv01 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn01 = nn.GroupNorm(8, self.in_channels)
#        self.dropout = nn.Dropout(self.drop)
        self.act = nn.ReLU()
        self.layer1 = self.make_layer(self.block, 64, self.layers[0], stride=1)
        self.layer2 = self.make_layer(self.block, 128, self.layers[1], stride=2)
        self.layer3 = self.make_layer(self.block, 256, self.layers[2], stride=2)
        self.layer4 = self.make_layer(self.block, 512, self.layers[3], stride=2)
        self.avg_pool = nn.AvgPool2d(6)
        self.fc = nn.Linear(512*self.block.expansion, self.num_class)
    

        self.apply(_weights_init)

        ### END YOUR CODE
    def make_layer(self, block, channels, num_blocks, stride):
        drop = self.drop
        strides = [stride] + [1]*(num_blocks-1)
        
        layers = []
        for stride in strides:
    #        print('make_layer', stride)
            layers.append(block(self.in_channels, channels, stride, drop=drop))
            self.in_channels = channels*block.expansion
    #        print("in_channels", self.in_channels)
        return nn.Sequential(*layers)




    def forward(self, x):
        '''
        Input x: a batch of images (batch size x 1 x 40 x 40)
        Return the predictions of each image (batch size x 6)
        '''
        ### YOUR CODE HERE
        out = x
        out = self.conv00(x)
        out = self.gn00(out)
 #       print(out.shape)
        out = self.conv01(out)
        out= self.gn01(out)
 #       out = self.dropout(out)
        out = self.act(out)
        out = self.layer1(out)
 #       print(out.shape)
        out = self.layer2(out)
 #       print(out.shape)
        out = self.layer3(out)
 #       print(out.shape)
        out = self.layer4(out)
 #       print(out.shape)
        out = self.avg_pool(out)
 #       print(out.shape)
        out = out.view(out.size(0), -1)
 #       print(out.shape)
        out = self.fc(out)

        ### END YOUR CODE
        return out
    

''' model candidates '''
def resnet8(num_class):
    return ResNet(blockBasic, [1, 1, 1, 1], num_class)


def resnet14(num_class):
    return ResNet(blockBasic, [2, 2, 2, 2], num_class)

def resnet20(num_class):
    return ResNet(blockBasic, [3, 3, 3, 3], num_class)

def resnet22(num_class):
    return ResNet(blockBasic, [3, 4, 3], num_class)

def resnet26(num_class):
    return ResNet(blockBasic, [4, 4, 4], num_class)

def resnet32(num_class):
    return ResNet(blockBasic, [5, 5, 5], num_class)

def resnet44(num_class):
    return ResNet(blockBasic, [7, 7, 7], num_class)

def resnet56(num_class):
    return ResNet(blockBasic, [9, 9, 9], num_class)

def test():
    net = resnet20()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())