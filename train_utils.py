import random
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
#import torchvision.datasets.ImageFolder
import torchvision.transforms as transforms
from data import Emotions
from torch.utils.data import DataLoader


#calculate accuracy
def accuracy(preds, labels):
    return (preds == labels).mean()


#load data
def load_data(batch_size):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
    ])
    data = Emotions(cvs_file='icml_face_data.csv', root_dir = './data', transform=transform)
    train_data, test_data = torch.utils.data.random_split(data, [0.8, 0.2])

    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader 


#train
def train(args, net, train_loader, optimizer, privacy_engine, epoch):
    device = next(net.parameters()).device
    net.train()
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    top1_acc = []
    for images, targets in train_loader:
        
        images, targets = images.to(args.device), targets.to(args.device)
        outputs = net(images)
        loss = criterion(outputs, targets)
        preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        labels = targets.detach().cpu().numpy()
        
        acc1 = accuracy(preds, labels)
        top1_acc.append(acc1)
        
        loss.backward()
        losses.append(loss.item())

        optimizer.step()
        optimizer.zero_grad()
    
    epsilon=None
    if not args.disable_dp:
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        print(
            f'Train Epoch: {epoch}\n'
            f'Train set: Average loss: {np.mean(losses):.4f}, '
            f'Acc@1: {100*np.mean(top1_acc):.2f}%, '
            f'(ε = {epsilon})'
        )
    else:
        print(
            f'Train Epoch: {epoch}\n'
            f'Train set: Average loss: {np.mean(losses):.4f}, '
            f'Acc@1: {100*np.mean(top1_acc):.2f}%'
        )
    return np.mean(losses), np.mean(top1_acc), epsilon
        
#        test(args, net, test_loader)

#test
def test(net, test_loader, device):
    device = next(net.parameters()).device
    net.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = net(images)
            loss = criterion(outputs, targets)
            preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            labels = targets.detach().cpu().numpy()
            acc1 = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc1)
        print(f'Test set: Average loss: {np.mean(losses):.4f}, '
              f'Acc@1: {100*np.mean(top1_acc):.2f}%'
            )
    return np.mean(top1_acc), np.mean(losses)

        

