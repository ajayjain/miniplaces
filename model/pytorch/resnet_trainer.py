#!/usr/bin/python

import os
import shutil
import time

from IPython.display import Image
# import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models 
import os.path

import DataLoader
# from densenet_modified import *
import sys

# Trainer parameters
print_freq_epochs = 100
use_cuda = True

# Dataset Parameters
batch_size = 32
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training parameters
# architecture = 'resnet34'
# architecture = 'vgg16_bn'
# architecture = 'dense'
lr = 0.1  # densenet default = 0.1, 
lr_init = 0.1
momentum = 0.90 # densenet default = 0.9 
weight_decay = 1e-3 # densenet default = 1e-4
num_epochs = 125

dummy_text_file = open("dummy_text.txt", "w")

def construct_dataloader_disk():
    # Construct DataLoader
    opt_data_train = {
        #'data_h5': 'miniplaces_128_train.h5',
        'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
        'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
        'load_size': load_size,
        'fine_size': fine_size,
        'data_mean': data_mean,
        'randomize': True
        }
    opt_data_val = {
        #'data_h5': 'miniplaces_128_val.h5',
        'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
        'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
        'load_size': load_size,
        'fine_size': fine_size,
        'data_mean': data_mean,
        'randomize': False
        }

    loader_train = DataLoader.DataLoaderDisk(**opt_data_train)
    loader_val = DataLoader.DataLoaderDisk(**opt_data_val)
    
    return (loader_train, loader_val)

def construct_dataloader_disk_trainval():
    opt_data_trainval = {
        #'data_h5': 'miniplaces_128_val.h5',
        'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
        'data_list': '../../data/trainval.txt',   # MODIFY PATH ACCORDINGLY
        'load_size': load_size,
        'fine_size': fine_size,
        'data_mean': data_mean,
        'randomize': False
        }
    loader_valtrain = DataLoader.DataLoaderDisk(**opt_data_trainval)
        
    return (loader_valtrain)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def adjust_learning_rate(lr, optimizer, epoch):
    """Calculates a learning rate of the initial LR decayed by 10 every 30 epochs"""
    lr = lr_init * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# def adjust_learning_rate(lr, optimizer, epoch): # for densenet (201)
#     """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
#     lr = lr_init * (0.1 ** (epoch // 20)) * (0.1 ** (epoch // 50))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def save_checkpoint(filename, model, state, is_best, epoch):
    torch.save(state, "models/"+filename) #"densenet121__retraining.tar"
    if is_best:
        torch.save(model, "results/"+filename)
        
# train and validate methods adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py

def train(train_loader, model, criterion, optimizer, epoch, text_file):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(int(train_loader.size()/batch_size)):
        input, target = train_loader.next_batch(batch_size)
        target = target.long()
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            target = target.cuda(async=True)
            input = input.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_var = target_var.long()
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
                
        if i % print_freq_epochs == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, train_loader.size()/batch_size, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            
    text_file.write(str(epoch)+str(",")+str(i)+str(",")+str(batch_time.val)+str(",")+str(data_time.val)+str(",")+str(losses.avg)+str(",")+str(top1.avg)+str(",")+str(top5.avg)+"\n")
        
def validate(val_loader, model, criterion, text_file, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i in range(int(val_loader.size()/batch_size)):
        input, target = val_loader.next_batch(batch_size)
        target = target.long()
        if use_cuda:
            target = target.cuda(async=True)
            input = input.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        target_var = target_var.long()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        
        if i % print_freq_epochs == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, val_loader.size()/batch_size, batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))


    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    
    text_file.write(str("val,")+str(epoch)+","+str(i)+str(",")+str(batch_time.val)+str(",")+str(losses.avg)+str(",")+str(top1.avg)+str(",")+str(top5.avg)+"\n")

    return top5.avg


criterion = nn.CrossEntropyLoss()

if use_cuda:
    criterion = criterion.cuda()
    
train_loader, val_loader = construct_dataloader_disk()
trainval_loader = construct_dataloader_disk_trainval()

def trainer(filename, lr, momentum, weight_decay):

#     filename = "resnet34"
   # model = models.__dict__[filename](num_classes=100, pretrained=False)
    model = torch.load("results/"+filename+".pt")

    if use_cuda:
        model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


    best_prec5 = 70.0

    text_file_train = open("results/"+filename+".txt", "w")
    text_file_val = open("results/"+filename+".txt", "w")

    for epoch in range(85,num_epochs):

        # check for file
        if not os.path.isfile(filename+".txt"):
            break

        lr = adjust_learning_rate(lr, optimizer, epoch) # turn off for Adam
        print("learning rate:", lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, text_file_train)

        # evaluate on validation set
        prec5 = validate(val_loader, model, criterion, text_file_val, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec5 > best_prec5
        best_prec5 = max(prec5, best_prec5)

        save_checkpoint(filename+".pt", model, {
            'epoch': epoch + 1,
            'arch': filename,
            'state_dict': model.state_dict(),
            'best_prec5': best_prec5,
            'optimizer' : optimizer.state_dict(),
        }, is_best, epoch)

    print("First round of training finished")
    model = torch.load("results/"+filename+".pt")

    filename_old = filename
    filename = filename+"_valtrained"

    text_file_train = open("results/"+filename+".txt", "w")
    text_file_val = open("results/"+filename+".txt", "w")

    print("Training on validation set:")

    for epoch in range(num_epochs,num_epochs+15):

        # check for file
        if not os.path.isfile(filename_old+".txt"):
            break

        lr = adjust_learning_rate(lr, optimizer, epoch) # turn off for Adam
        print("learning rate:", lr) # questionable

        # train for one epoch
        train(trainval_loader, model, criterion, optimizer, epoch, text_file_train)

        # evaluate on validation set
        prec5 = validate(val_loader, model, criterion, text_file_val, epoch) # pointless

        # remember best prec@1 and save checkpoint
        is_best = prec5 > best_prec5
        best_prec5 = max(prec5, best_prec5)

        save_checkpoint(filename+".pt", model, {
            'epoch': epoch + 1,
            'arch': filename,
            'state_dict': model.state_dict(),
            'best_prec5': best_prec5,
            'optimizer' : optimizer.state_dict(),
        }, is_best, epoch)
    return 0

trainer(str(sys.argv[1]),lr, momentum, weight_decay)
