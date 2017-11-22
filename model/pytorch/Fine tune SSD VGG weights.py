
# coding: utf-8

# In[22]:

get_ipython().magic('matplotlib inline')

import math
import os
import shutil
import string
import sys
import time

sys.path.append("../../ssd/ssd.pytorch/")

from IPython.display import Image
import matplotlib.pyplot as plt

import numpy as np
import scipy
import scipy.misc
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


# In[3]:

# Trainer parameters
print_freq_epochs = 5
use_cuda = True

# Dataset Parameters
batch_size = 50
# load_size = 342
# fine_size = 300
load_size = 300
fine_size = 300
c = 3
# data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])
data_mean = np.asarray([0.0, 0.0, 0.0])
num_classes = 100

# Training parameters
architecture = 'vggbn'
lr = 0.0005  # densenet default = 0.1, 
lr_init = 0.0005
momentum = 0.9 # densenet default = 0.9 
weight_decay = 1e-3 # densenet default = 1e-4
num_epochs = 45


# In[4]:

# Loading data from disk
class DataLoaderDisk(object):
    def __init__(self, **kwargs):
        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']
        self.data_root = os.path.join(kwargs['data_root'])

        # read data info from lists
        self.list_im = []
        self.list_lab = []
        with open(kwargs['data_list'], 'r') as f:
            for line in f:
                path, lab =line.rstrip().split(' ')
                self.list_im.append(os.path.join(self.data_root, path))
                self.list_lab.append(int(lab))
        self.list_im = np.array(self.list_im, np.object)
        self.list_lab = np.array(self.list_lab, np.int64)
        self.num = self.list_im.shape[0]
        print('# Images found:', self.num)

        # permutation
        perm = np.random.permutation(self.num) 
        self.list_im[:, ...] = self.list_im[perm, ...]
        self.list_lab[:] = self.list_lab[perm, ...]

        self._idx = 0
        
    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3), )
        labels_batch = np.zeros(batch_size, dtype=np.double)
        for i in range(batch_size):
            image = scipy.misc.imread(self.list_im[self._idx])
            image = scipy.misc.imresize(image, (self.load_size, self.load_size))
            image = image.astype(np.float32)/255.
            image = image - self.data_mean
            if self.randomize:
                flip = np.random.random_integers(0, 1)
                if flip>0:
                    image = image[:,::-1,:]
                offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
            else:
                offset_h = (self.load_size-self.fine_size)//2
                offset_w = (self.load_size-self.fine_size)//2

            images_batch[i, ...] =  image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
            labels_batch[i, ...] = self.list_lab[self._idx]
            
            self._idx += 1
            if self._idx == self.num:
                self._idx = 0
        
        # Switch to NCHW ordering and convert to torch FloatTensor
        images_batch = torch.from_numpy(images_batch.swapaxes(2, 3).swapaxes(1, 2)).float()
        labels_batch = torch.from_numpy(labels_batch).long()
        return images_batch, labels_batch
    
    def size(self):
        return self.num

    def reset(self):
        self._idx = 0
        
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

    loader_train = DataLoaderDisk(**opt_data_train)
    loader_val = DataLoaderDisk(**opt_data_val)

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
    loader_valtrain = DataLoaderDisk(**opt_data_trainval)

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
    lr = lr_init * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

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


# # Define and load VGG model

# In[5]:

class VGG(nn.Module):
    def __init__(self, base_vgg_layers, num_classes=100):
        super(VGG, self).__init__()
        self.base_vgg = base_vgg
        self.layers = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def features(self, x):
        for layer in self.base_vgg:
            x = layer(x)

        for layer in self.layers:
            x = layer(x)

        return x
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Conv2d):
                print("Initializing", m)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

#         m = self.classifier[0]
#         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#         m.weight.data.normal_(0, math.sqrt(2. / n))
#         if m.bias is not None:
#             m.bias.data.zero_()
            
        for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             el
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# From ssd.py
# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, in_channels, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

vgg_base_configuration = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]


# In[6]:

ssd_vgg_weights = {}
for key, weights in torch.load("../../ssd/ssd.pytorch/weights/ssd300_MiniPlaces_0.pth").items():
    # Find weights with keys prefixed by "vgg.", and remove prefix
    if key[:4] == 'vgg.':
        ssd_vgg_weights[key[4:]] = weights

base_vgg = nn.ModuleList(vgg(vgg_base_configuration, in_channels=3))
base_vgg.load_state_dict(ssd_vgg_weights)


# In[7]:

model = VGG(base_vgg, num_classes=num_classes)

if use_cuda:
    model = model.cuda(device_id=0)


# In[8]:

model


# # Fine tune

# In[25]:

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
            
            with open(text_file, "a+") as f:
                f.write('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                   epoch, i, train_loader.size()/batch_size, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            
    with open(text_file, "a+") as f:
        f.write(str(epoch)+str(",")+str(i)+str(",")+str(batch_time.val)+str(",")+str(data_time.val)+str(",")+str(losses.avg)+str(",")+str(top1.avg)+str(",")+str(top5.avg)+"\n")
        
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

            with open(text_file, "a+") as f:
                f.write('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, val_loader.size()/batch_size, batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))


    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    
    with open(text_file, "a+") as f:
        f.write(str("val,")+str(epoch)+","+str(i)+str(",")+str(batch_time.val)+str(",")+str(losses.avg)+str(",")+str(top1.avg)+str(",")+str(top5.avg)+"\n")

    return top5.avg


# In[10]:

criterion = nn.CrossEntropyLoss()

if use_cuda:
    criterion = criterion.cuda(device_id=0)

# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
optimizer = torch.optim.Adam(model.parameters())


# In[11]:

train_loader, val_loader = construct_dataloader_disk()
trainval_loader = construct_dataloader_disk_trainval()


# In[ ]:

best_prec5 = 0.0

filename = "vgg_ssd_lr5E-4_bn"
text_file_train = "results/"+filename+".txt"
text_file_val = "results/"+filename+".txt"


# In[ ]:

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


# In[ ]:

for epoch in range(1, num_epochs):
    # lr = adjust_learning_rate(lr, optimizer, epoch) # turn off for Adam
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

