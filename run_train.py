import os

name = 'test'
epoch = 200
batch = 64
nc = 2
lr = 0.001
lr_lambda = 0.8
img_size = 224
validation_rate = 0.1
train_path = r'./dataset/sample/train/'
val_path = r'./dataset/sample/val/'
model  = 0
loss = 'ce'
ocmd = r'python train.py --name {} --model {} --epoch {} --train_path {} --val_path {} --bs {} --nc {} --lr {} --img_size {} --vrate {} --loss {} --lr_lambda {}'

cmd = ocmd.format(name, model, epoch, train_path, val_path, batch, nc, lr, img_size, validation_rate, loss, lr_lambda)
print(cmd)
os.system(cmd)
