import argparse
import torch
from train_utils import load_data, train, test
from ResNet4_dp import resnet8, resnet14, resnet20
from models import CNN
from opacus import PrivacyEngine
from torch import optim
from log import Logger
import numpy as np




def get_args():
    parser = argparse.ArgumentParser(description='Args for training networks')
    parser.add_argument('-num_classes', type=int, default=7, help='number of classes')
    parser.add_argument('-epochs', type=int, default=50, help='num epochs')
    parser.add_argument('-batch', type=int, default=512, help='batch size')
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-net', type=str, default ='resnet8', help="net type choose from []")
    parser.add_argument('-delta', type=int, default=1e-5, help='privacy parameter delta')
    parser.add_argument('-disable-dp', type=bool, default=False, help='disable differential private training')
    parser.add_argument('-logdir', type=str, default='./log/', help='Where the tensorboard will be stored')
    parser.add_argument('-device', type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), help='Device on which to run')
    parser.add_argument('-sigma', type=float, default=1.1, help='Noise multiplier in DPSGD')
    parser.add_argument('-C', type=float, default=0.1, help='Max gradient bound in DPSGD')
    parser.add_argument('-lr-schedule', type=str, choices=["constant", "cos"], default="constant")
    args, _ = parser.parse_known_args()
    return args

def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    args = get_args()
    
    train_loader, test_loader = load_data(args.batch)
    if args.net == "resnet8":  
        net = resnet8(args.num_classes)
    elif args.net == "resnet14":
        net = resnet14(args.num_classes)
    elif args.net == "resnet20":
        net = resnet20(args.num_classes)
    else:
        net = CNN()
    net = net.to(args.device)
 #   print(net)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    
    privacy_engine = None
    if not args.disable_dp: 
        logger = Logger(f'{args.logdir}/{args.C}_{args.sigma}_{args.lr_schedule}')
        privacy_engine = PrivacyEngine()
        net, optimizer, train_loader = privacy_engine.make_private(
        module=net,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=args.sigma,
        max_grad_norm=args.C,
        )
    else:
        logger = Logger(f'{args.logdir}/{args.lr}_{args.batch}_{args.net}')
    print(f'The number of parameters: {get_num_params(net)}')

    for epoch in range(1, args.epochs+1):
        if args.lr_schedule == "cos":
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / (args.epochs+1)))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        train_loss, train_acc, epsilon= train(args, net, train_loader, optimizer, privacy_engine, epoch)
        top1_acc, test_loss = test(net, test_loader, args.device)

        logger.log_epoch(epoch, train_loss, train_acc, test_loss, top1_acc, epsilon)
        logger.log_scalar("epsilon/train", epsilon, epoch)

if __name__ == '__main__':
    main()

