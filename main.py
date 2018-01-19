import os, time, shutil, argparse
from functools import partial
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from skimage import io

from utils import save_checkpoint, AverageMeter, visualize_image, GrayscaleImageFolder
from model import ColorNet

# Parse arguments and prepare program
parser = argparse.ArgumentParser(description='Training and Using ColorNet')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to .pth file checkpoint (default: none)')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (overridden if loading from checkpoint)')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='size of mini-batch (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='learning rate at start of training')
parser.add_argument('--weight-decay', '--wd', default=1e-10, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='use this flag to validate without training')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')

# Current best losses
best_losses = 1000.0
use_gpu = torch.cuda.is_available()

def main():
    global args, best_losses, use_gpu
    args = parser.parse_args()
    print('Arguments: {}'.format(args))
    
    # Create model  # models.resnet18(num_classes=365)
    model = ColorNet()
    
    # Use GPU if available
    if use_gpu:
        model.cuda()
        print('Loaded model onto GPU.')
    
    # Create loss function, optimizer #criterion = nn.CrossEntropyLoss().cuda() if use_gpu else nn.CrossEntropyLoss()
    criterion = nn.MSELoss().cuda() if use_gpu else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('Loading checkpoint {}...'.format(args.resume))
            checkpoint = torch.load(args.resume) if use_gpu else torch.load(args.resume, map_location=lambda storage, loc: storage)
            args.start_epoch = checkpoint['epoch']
            best_losses = checkpoint['best_losses']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Finished loading checkpoint. Resuming from epoch {}'.format(checkpoint['epoch']))
        else:
            print('Checkpoint filepath incorrect.')
            return
    elif args.pretrained:
        print(args.pretrained)
        if os.path.isfile(args.pretrained):
            model.load_state_dict(torch.load(args.pretrained))
            print('Loaded pretrained model.')
        else:
            print('Pretrained model filepath incorrect.')
            return
    
    # Load data from pre-defined (imagenet-style) structure
    if not args.evaluate:
        train_directory = os.path.join(args.data, 'train')
        train_transforms = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip()
        ])
        train_imagefolder = GrayscaleImageFolder(train_directory, train_transforms)
        train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        print('Loaded training data.')
    val_transforms = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224)
    ])
    val_directory = os.path.join(args.data, 'val')
    val_imagefolder = GrayscaleImageFolder(val_directory , val_transforms)
    val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print('Loaded validation data.')

    # If in evaluation (validation) mode, do not train
    if args.evaluate:
        save_images = True
        epoch = 0
        initial_losses = validate(val_loader, model, criterion, save_images, epoch)

        # # Save checkpoint after evaluation if desired
        # save_checkpoint({
        #     'epoch': epoch,
        #     'best_losses': initial_losses,
        #     'state_dict': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }, False, 'checkpoints/evaluate-checkpoint.pth.tar')
        
        return  
    
    # Otherwise, train for given number of epochs
    validate(val_loader, model, criterion, False, 0) # validate before training
    for epoch in range(args.start_epoch, args.epochs):
        
        # Train for one epoch, then validate
        train(train_loader, model, criterion, optimizer, epoch)
        save_images = True #(epoch % 3 == 0)
        losses = validate(val_loader, model, criterion, save_images, epoch)
        
        # Save checkpoint, and replace the old best model if the current model is better
        is_best_so_far = losses < best_losses
        best_losses = max(losses, best_losses)
        save_checkpoint({
            'epoch': epoch + 1,
            'best_losses': best_losses,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best_so_far, 'checkpoints/checkpoint-epoch-{}.pth.tar'.format(epoch))
        
    return best_losses

def train(train_loader, model, criterion, optimizer, epoch):
    '''Train model on data in train_loader for a single epoch'''
    print('Starting training epoch {}'.format(epoch))

    # Prepare value counters and timers
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    # Switch model to train mode
    model.train()
    
    # Train for single eopch
    end = time.time()
    for i, (input_gray, input_ab, target) in enumerate(train_loader):
        
        # Use GPU if available
        input_gray_variable = Variable(input_gray).cuda() if use_gpu else Variable(input_gray)
        input_ab_variable = Variable(input_ab).cuda() if use_gpu else Variable(input_ab)
        target_variable = Variable(target).cuda() if use_gpu else Variable(target)

        # Record time to load data (above)
        data_time.update(time.time() - end)

        # Run forward pass
        output_ab = model(input_gray_variable) # throw away class predictions
        loss = criterion(output_ab, input_ab_variable) # MSE
        
        # Record loss and measure accuracy
        losses.update(loss.data[0], input_gray.size(0))
        
        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record time to do forward and backward passes
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print model accuracy -- in the code below, val refers to value, not validation
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses)) 

    print('Finished training epoch {}'.format(epoch))

def validate(val_loader, model, criterion, save_images, epoch):
    '''Validate model on data in val_loader'''
    print('Starting validation.')

    # Prepare value counters and timers
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    # Switch model to validation mode
    model.eval()
    
    # Run through validation set
    end = time.time()
    for i, (input_gray, input_ab, target) in enumerate(val_loader):
        
        # Use GPU if available
        target = target.cuda() if use_gpu else target
        input_gray_variable = Variable(input_gray, volatile=True).cuda() if use_gpu else Variable(input_gray, volatile=True)
        input_ab_variable = Variable(input_ab, volatile=True).cuda() if use_gpu else Variable(input_ab, volatile=True)
        target_variable = Variable(target, volatile=True).cuda() if use_gpu else Variable(target, volatile=True)
       
        # Record time to load data (above)
        data_time.update(time.time() - end)

        # Run forward pass
        output_ab = model(input_gray_variable) # throw away class predictions
        loss = criterion(output_ab, input_ab_variable) # check this!
        
        # Record loss and measure accuracy
        losses.update(loss.data[0], input_gray.size(0))

        # Save images to file
        if save_images:
            for j in range(len(output_ab)):
                save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
                save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
                visualize_image(input_gray[j], ab_input=output_ab[j].data, show_image=False, save_path=save_path, save_name=save_name)

        # Record time to do forward passes and save images
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print model accuracy -- in the code below, val refers to both value and validation
        if i % args.print_freq == 0:
            print('Validate: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses))

    print('Finished validation.')
    return losses.avg

if __name__ == '__main__':
    main()
