## --------------------------------------------------------------------
##         B0    B1      B2      B3      B4      B5      B6      B7
## ====================================================================
## Input   224   240     260     300     380     456     528     600
## output  1280  1280    1408    1536    1792    2048    2304    2560
## --------------------------------------------------------------------
##

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import copy
import random

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from efficientnet_pytorch import EfficientNet

from torchvision import transforms, datasets
from torchvision.utils import save_image

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', help='running name', default='twim')
parser.add_argument('--model', help='model number 0 - 7', default='5')
parser.add_argument('--epoch', help='epoch', type=int, default=100)
parser.add_argument('--train_path', help='epoch', type=str, default='./dataset')
parser.add_argument('--val_path', help='epoch', type=str, default=None)
parser.add_argument('--bs', help='batch size', type=int, default=8)
parser.add_argument('--nc', help='number of classes', type=int, default=1)
parser.add_argument('--lr', help='learning rate', type=float, default=0.01)
parser.add_argument('--wd', help='weight decay', type=float, default=1e-5)
parser.add_argument('--img_size', help='image size', type=int, default=200)
parser.add_argument('--vrate', help='validation rate', type=float, default=0.1)
parser.add_argument('--optim', help='adam | sgd', type=str, default='adam')
parser.add_argument('--gpu', help='gpu device number', type=int, default=0)
parser.add_argument('--loss', help='loss function (bce | ce | msm)', type=str, default='ce')
parser.add_argument('--lr_lambda', help='learning rate lambda', type=float, default=0.98739)

parser.add_argument('--clr_b', help='ColorJitter brightness', type=float, default=0.2)
parser.add_argument('--clr_c', help='ColorJitter contrast', type=float, default=0.2)
parser.add_argument('--clr_s', help='ColorJitter saturation', type=float, default=0.2)
parser.add_argument('--clr_h', help='ColorJitter hue', type=float, default=0.2)
parser.add_argument('--hflip', help='Random Horizontal Flip', type=float, default=0.5)
parser.add_argument('--vflip', help='Random Vertical Flip', type=float, default=0.5)
parser.add_argument('--rotate', help='Random Rotation (degree)', type=int, default=0)

opt = parser.parse_args()

testname = opt.name

data_path = opt.train_path
test_path = opt.val_path

# fc 제외하고 freeze
# for n, p in model.named_parameters():
#     if '_fc' not in n:
#         p.requires_grad = False
# model = torch.nn.parallel.DistributedDataParallel(model)

#########################################################################################################
## parameters for dataloader
#########################################################################################################
batch_size  = opt.bs
random_seed = 555
random.seed(random_seed)
torch.manual_seed(random_seed)

#########################################################################################################
## train dataset
#########################################################################################################
president_dataset = datasets.ImageFolder(
                                data_path,
                                transforms.Compose([
                                    transforms.Resize((opt.img_size, opt.img_size)),
                                    transforms.ToTensor(),
                                    transforms.ColorJitter(brightness=opt.clr_b, contrast=opt.clr_c, saturation=opt.clr_s, hue=opt.clr_h),
                                    transforms.RandomHorizontalFlip(p=opt.hflip),
                                    transforms.RandomVerticalFlip(p=opt.vflip),
                                    transforms.RandomRotation(degrees=opt.rotate),
                                    # transforms.RandomResizedCrop(size=512, scale=(0.08, 1.0), ratio=(0.75, 1.33), interpolation=2),
                                    # transforms.RandomErasing(),
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ]))

#########################################################################################################
## validation dataset
#########################################################################################################
datasets_dict = {}
if test_path is not None:
    datasets_dict['train'] = president_dataset
    datasets_dict['valid'] = datasets.ImageFolder(
                                test_path,
                                transforms.Compose([
                                    transforms.Resize((opt.img_size, opt.img_size)),
                                    transforms.ToTensor(),
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

                                    transforms.ColorJitter(brightness=opt.clr_b, contrast=opt.clr_c, saturation=opt.clr_s, hue=opt.clr_h),
                                    transforms.RandomHorizontalFlip(p=opt.hflip),
                                    transforms.RandomVerticalFlip(p=opt.vflip),
                                    transforms.RandomRotation(degrees=opt.rotate),
                                ]))
else:
    train_idx, tmp_idx = train_test_split(list(range(len(president_dataset))), test_size=opt.vrate, random_state=random_seed)
    datasets_dict['train'] = Subset(president_dataset, train_idx)
    datasets_dict['valid'] = Subset(president_dataset, tmp_idx)

#########################################################################################################
## define data loader
#########################################################################################################
dataloaders, batch_num = {}, {}
dataloaders['train'] = torch.utils.data.DataLoader(datasets_dict['train'],
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=0)
dataloaders['valid'] = torch.utils.data.DataLoader(datasets_dict['valid'],
                                              batch_size=1, shuffle=False,
                                              num_workers=0)

batch_num['train'], batch_num['valid'] = len(dataloaders['train']), len(dataloaders['valid'])
print('batch_size : %d,  tvt : %d / %d' % (batch_size, batch_num['train'], batch_num['valid']))

# for i, (inputs, labels) in enumerate(dataloaders['test']):
#     print(labels)
#     sample_fname, a = dataloaders['test'].dataset.samples[i]
#     print(sample_fname, a)

#########################################################################################################
## model
#########################################################################################################
device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")  # set gpu

model_name = f'efficientnet-b{opt.model}'
image_size = EfficientNet.get_image_size(model_name)
print('model input size: ', image_size)
model = EfficientNet.from_pretrained(model_name, num_classes=opt.nc)
model = model.to(device)

#########################################################################################################
## parameters
#########################################################################################################
epoch = opt.epoch

if opt.loss == 'ce':
    criterion = nn.CrossEntropyLoss()
    
elif opt.loss == 'msm':
    criterion = nn.MultiLabelSoftMarginLoss()

if opt.optim == 'sgd':
    optimizer_ft = optim.SGD(model.parameters(), 
                             lr = opt.lr,
                             momentum=0.9,
                             weight_decay=1e-4)
elif opt.optim == 'adam':
    optimizer_ft = optim.Adam(model.parameters(), lr = opt.lr, weight_decay=opt.wd)

lmbda = lambda epoch: opt.lr_lambda
exp_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer_ft, lr_lambda=lmbda)

##########################################################################################
### make folders
##########################################################################################
# tbroot = './tensorboard'
# if not os.path.exists(tbroot):
#      os.makedirs(tbroot)
# tbroot = tbroot + '/' + testname
# if not os.path.exists(tbroot):
#     os.makedirs(tbroot)
# twriter = SummaryWriter(tbroot)

exp_root = './exp'
if not os.path.exists(exp_root):
     os.makedirs(exp_root)

train_root = f'{exp_root}/train'
if not os.path.exists(train_root):
     os.makedirs(train_root)

train_root = train_root + '/' + testname
if not os.path.exists(train_root):
    os.makedirs(train_root)

twriter = SummaryWriter(train_root)

##########################################################################################
### save arguments
##########################################################################################
opt_list = [ f'{key} : {opt.__dict__[key]}' for key in opt.__dict__ ]
with open(f'{train_root}/args.txt', 'w') as f:
    [f.write(f'{st}\n') for st in opt_list]


##########################################################################################
### train
##########################################################################################
softmax = nn.Softmax(dim=1)
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss, running_corrects, num_cnt = 0.0, 0, 0
            

            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                tepoch.set_description(f"[{phase}] Epoch {epoch}")

                # Iterate over data.
                for inputs, labels in tepoch:

                    ## RGB to BGR
                    # inputs = inputs[:,[2,1,0],:]
                    
                    inputs = inputs.to(device)
                    # labels = labels.type(torch.FloatTensor)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        
                        # if opt.nc == 2:
                        # outputs = softmax(outputs)

                        _, preds = torch.max(outputs, 1)
                        
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    num_cnt += len(labels)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = float(running_loss / num_cnt)
                epoch_acc  = float((running_corrects.double() / num_cnt).cpu()*100)
                
                tepoch.set_postfix(loss=loss.item(), accuracy=epoch_acc)

                if phase == 'train':
                    train_loss.append(epoch_loss)
                    train_acc.append(epoch_acc)

                    twriter.add_scalar("train/loss", epoch_loss, epoch)
                    twriter.add_scalar("train/acc", epoch_acc, epoch)

                else:
                    valid_loss.append(epoch_loss)
                    valid_acc.append(epoch_acc)

                    twriter.add_scalar("val/loss", epoch_loss, epoch)
                    twriter.add_scalar("val/acc", epoch_acc, epoch)

                print('{} Loss: {:.4f} Acc: {:.2f}'.format(phase, epoch_loss, epoch_acc))
               
                # deep copy the model
                if phase == 'valid' and epoch_acc > best_acc:
                    best_idx = epoch
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), f'{train_root}/best.pt')
                    print('==> best model saved - %d / %.1f'%(best_idx, best_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc (Epoch %d): %.1f' %(best_idx, best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), f'{model_name}-{testname}-b{batch_size}_final.pt')
    print('model saved')
    return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc

if __name__ == "__main__":

    model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc = train_model(
        model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epoch)


    

