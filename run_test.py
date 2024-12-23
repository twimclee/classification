import os

name = 'noul_cont_test-20230202'

nc = 2

img_size = 528

data_path = r'D:\data\noul\inplace\2023\30d\ng\Image\\cropped'
# data_path = r'D:\data\noul\inplace\2023\30d_test\Image\\cropped'

weight_path = r'F:\projects\NOUL\models\20230202\noul_cont.pt'

class_list = r'0.ok 1.ng'

model = 6

subdir = True

ocmd = r'python test.py --name {} --model {} --data_path {} --weight_path {} --nc {} --img_size {} --class_list {} --subdir {}'

cmd = ocmd.format(name, model, data_path, weight_path, nc, img_size, class_list, subdir)
print(cmd)
os.system(cmd)


