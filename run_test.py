import os

name = 'test'

nc = 2

img_size = 224

data_path = r'./dataset/sample/val/'

weight_path = r'./exp/train/test/best.pt'

class_list = r'0.ok 1.ng'

model = 0

subdir = True

ocmd = r'python test.py --name {} --model {} --data_path {} --weight_path {} --nc {} --img_size {} --class_list {} --subdir {}'

cmd = ocmd.format(name, model, data_path, weight_path, nc, img_size, class_list, subdir)
print(cmd)
os.system(cmd)


