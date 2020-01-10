'''
Train network using pytorch

author Gu Jiapan
'''

import argparse
import torch
import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models

from ReadDataSource import ReadDataSource
# from ResNet import ResNet
from Model import Model
# from SE_block import SEBottleneck


# def se_resnet50():
#     model = ResNet(SEBottleneck, [3,4,6,3])
#     model.avgpool = nn.AdaptiveAvgPool2d(1)
#     return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_dir', type=str, default='/data/image')
    parser.add_argument('--label_file', type=str, default='/data/label.csv')
    args = parser.parse_args()

    train_dataset = ReadDataSource(x_dir=args.image_dir, y_file=args.label_file)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = ReadDataSource(x_dir=args.image_dir, y_file=args.label_file)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

    model = Model()
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(args.num_epoch):
        losses = 0.
        count = 0
        model.train()

        for idx, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda() 
            outs = model(x)

            loss = criterion(outs, y)
            losses += loss.item()*x.size(0)
            count += x.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch {:3d} avg lose {:10.6f}'.format(epoch, losses/count))

        val_loss = 0.
        count = 0
        correct = 0
        model.eval()
        for idx, (x, y) in enumerate(test_loader):
            x, y = x.cuda(), y.cuda() 
            outs = model(x)

            loss = criterion(outs, y)
            val_loss += loss.item()*x.size(0)
            count += x.size(0)
            _, preds = outs.max(1)
            correct += preds.eq(y).sum()

        acc = correct.float() / count

        print('Check! acc {:10.6f}  avg lose {:10.6f}'.format(acc, val_loss / count))


if __name__ == '__main__':
    main()