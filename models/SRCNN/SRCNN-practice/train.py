import argparse
import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import SRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr

if __name__ == '__main__':
    root = '/root/to/data'
    train_file = os.path.join(root, '91-image_x3.h5')
    eval_file = os.path.join(root, 'Set5_x3.h5')
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type = str, default = train_file)
    parser.add_argument('--eval-file', type = str, default = eval_file)
    parser.add_argument('--outputs-dir', type = str, default = './outputs')
    parser.add_argument('--scale', type = int, default = 3)
    parser.add_argument('--lr', type = float, defalut = 1e-4)
    parser.add_argument('--batch-size', type = int, default = 16)
    parser.add_argument('--num-epochs', type = int, default = 400)
    parser.add_argument('--num_workers', type = int, default = 8)
    parser.add_argument('--seed', type = int, default = 123)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, f'x{args.scale}')

    os.makedirs(args.outputs_dir, exist_ok = True)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()}, 
        {'params': model.conv2.parameters()}, 
        {'params': model.conv3.parameters(), 
         'lr': args.lr * 0.1}
    ], lr = args.lr)

    train_dataset = TrainDataset(h5_file = args.train_file)
    train_dataloader = DataLoader(dataset = train_dataset, 
                                  batch_size = args.batch_size, 
                                  shuffle = True, 
                                  num_workers = args.num_workers, 
                                  pin_memory = True, 
                                  drop_last = True,)
    
    eval_dataset = EvalDataset(h5_file = args.eval_file)
    eval_dataloader = DataLoader(dataset = eval_dataset, 
                                 batch_size = 1,)
    
    best_weights = copy.deepcopy(x = model.state_dict())
    best_epoch = 0
    best_psnr = 0.

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total = (len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description(f'epoch: {epoch} / {args.num_epochs - 1}')

            for data in train_dataloader:
                inputs, lables = data
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                t.set_postfix(loss = '{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        torch.save(obj = model.state_dict(), 
                   f = os.path.join(args.outputs_dir, f'epoch_{epoch + 1}.pth'))
        
        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)
                
            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print(f'eval_psnr: {epoch_psnr.ave:.2f}')

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch + 1
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

        print(f'best epoch: {best_epoch + 1}, psnr: {best_psnr:.2f}')
        torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))