from efficientnet_pytorch import EfficientNet

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.utils import save_image

from PIL import Image

import numpy as np
import os

import argparse

import shutil

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
parser.add_argument('--gpu', type=int, default=0, help='gpu device number')
opt = parser.parse_args()

#################################################################################################
### load model
#################################################################################################
model_name = f'efficientnet-b{opt.model}'
model = EfficientNet.from_pretrained(model_name, num_classes=opt.nc)
model.load_state_dict(torch.load(opt.weight_path))
model.eval()

#################################################################################################
### load data
#################################################################################################
test_dataset = datasets.ImageFolder(
                                opt.data_path,
                                transforms.Compose([
                                    transforms.Resize((opt.img_size, opt.img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ]))

dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")  # set gpu
model = model.to(device)

criterion = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=1)

############################################################################################
## make folders to save test results
############################################################################################
exp_root = './exp'
if not os.path.exists(exp_root):
    os.makedirs(exp_root)

val_root = f'{exp_root}/val'
if not os.path.exists(val_root):
    os.makedirs(val_root)

val_root = f'{val_root}/{opt.name}'
if not os.path.exists(val_root):
    os.makedirs(val_root)

val_suc_root = f'{val_root}/suc'
if not os.path.exists(val_suc_root):
    os.makedirs(val_suc_root)

val_fail_root = f'{val_root}/fail'
if not os.path.exists(val_fail_root):
    os.makedirs(val_fail_root)

############################################################################################
## test and visulaize
############################################################################################
def test_and_save(model):
    model.eval()

    running_loss, running_corrects, num_cnt = 0.0, 0, 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            fname, _ = dataloader.dataset.samples[i]
            ext = fname.split('.')[-1]

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            if opt.nc == 2:
                outputs = softmax(outputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)  # batch의 평균 loss 출력

            running_loss    += loss.item() * inputs.size(0)
            running_corrects+= torch.sum(preds == labels.data)
            num_cnt += inputs.size(0)  # batch size

            # ln = labels[0].cpu().numpy()
            l = opt.class_list[labels[0].cpu().numpy()]
            p = opt.class_list[preds[0].cpu().numpy()]
            if labels[0].cpu().numpy() == preds[0].cpu().numpy():
                if not os.path.exists(f'{val_suc_root}/{l}'):
                    os.makedirs(f'{val_suc_root}/{l}')
                shutil.copyfile(fname, f'{val_suc_root}/{l}/{i}_{l}_{p}.{ext}')
            else:
                if not os.path.exists(f'{val_fail_root}/{l}'):
                    os.makedirs(f'{val_fail_root}/{l}')
                shutil.copyfile(fname, f'{val_fail_root}/{l}/{i}_{l}_{p}.{ext}')

        test_loss = running_loss / num_cnt
        test_acc  = running_corrects.double() / num_cnt       
        print('val done : loss/acc : %.2f / %.2f' % (test_loss, test_acc*100))

if __name__ == "__main__":

    test_and_save(model)