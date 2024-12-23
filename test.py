from efficientnet_pytorch import EfficientNet

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.utils import save_image
import cv2

from dataset import ImageDataSet

import numpy as np
import os

import argparse

import shutil
import time


#################################################################################################
### parameters
#################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--name', help='running name', default='twim')
parser.add_argument('--model', help='model number 0 - 7', default='0')
parser.add_argument('--data_path', help='test dataset path', type=str, default='./dataset')
parser.add_argument('--weight_path', help='weight path', type=str, default='./now.pt')
parser.add_argument('--nc', help='number of classes', type=int, default=1)
parser.add_argument('--img_size', help='image size', type=int, default=200)
parser.add_argument('--class_list', nargs='+', default=['defect'], help='class list')
parser.add_argument('--device', type=int, default=0, help='device number')
parser.add_argument('--subdir', type=bool, default=False, help='load all files in subdiretory')
opt = parser.parse_args()

#################################################################################################
### load model
#################################################################################################
model_name = f'efficientnet-b{opt.model}'
model = EfficientNet.from_pretrained(model_name, num_classes=opt.nc)
model.load_state_dict(torch.load(opt.weight_path))
model.eval()

softmax = nn.Softmax(dim=1)

#################################################################################################
### load data
#################################################################################################
test_dataset = ImageDataSet(
    root = opt.data_path,
    transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    subdir = opt.subdir)

if len(test_dataset) == 0: exit()

dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")  # set gpu
model = model.to(device)

############################################################################################
## make folders to save test results
############################################################################################
exp_root = './exp'
if not os.path.exists(exp_root):
    os.makedirs(exp_root)

test_root = f'{exp_root}/test'
if not os.path.exists(test_root):
    os.makedirs(test_root)

test_root = f'{test_root}/{opt.name}'
if not os.path.exists(test_root):
    os.makedirs(test_root)

############################################################################################
## warm up1
[]############################################################################################
for i, (img, path) in enumerate(dataloader):
    img = img.to(device)
    model(img)
    if i == 5:
        break

############################################################################################
## test and save
############################################################################################+
def test_and_save(model):
    model.eval()

    total = len(test_dataset)
    stime = time.time()
    with torch.no_grad():
        
        for i, (img, path) in enumerate(dataloader):
            fullname = path[0]
            fname = fullname.split('\\')[-1]

            # save_image(img[0], 'img1.png')

            print(img.shape)

            ## RGB to BGR
            # img = img[:,[2,1,0],:]

            img = img.to(device)

            outputs = model(img)
            if opt.nc == 2:
                outputs = softmax(outputs)
            _, preds = torch.max(outputs, 1)

            outputs = outputs.cpu().detach().numpy()
            index = preds[0].cpu().numpy()
            p = opt.class_list[index]
            print(f'[{i}/{total}]: {fname} => {p} {outputs[0][index]*100}%')
            if not os.path.exists(f'{test_root}/{p}'):
                os.makedirs(f'{test_root}/{p}')
            
            # if index == 1:
            shutil.copyfile(fullname, f'{test_root}/{p}/{fname}')

    etime = time.time()
    ttime = etime - stime
    ttl_time = round(ttime, 2)
    avg_time = round(ttime/(i+1) * 1000, 2)
    print(f"[time] total: {ttl_time}sec, avg: {avg_time}ms")


if __name__ == "__main__":
    test_and_save(model)


