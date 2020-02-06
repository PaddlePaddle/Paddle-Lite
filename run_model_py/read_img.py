# coding=utf-8 
from PIL import Image
import sys
import os
import glob
# python 函数
# 功 能：将一张 jpg文件转pgm格式文件
# 参 数：jpg_file : 欲转换的jpg文件名
#              pgm_dir  : 存放 pgm 文件的目录
def jpg2pgm(jpg_path, pgm_path):
    # 首先打开jpg文件
    jpg = Image.open(jpg_path)
    # resize 双线性插值
    jpg = jpg.resize((1080, 1920), Image.BILINEAR)
    # 调用 python 函数 os.path.join , os.path.splitext , os.path.basename ，产生目标pgm文件名
    name = (str)(os.path.join(pgm_path, os.path.splitext(os.path.basename(jpg_path))[0])) + ".pgm"
    # 创建目标pgm 文件
    print('name: ', name)
    jpg.save(name)

 
if __name__ == '__main__':
    jpg = Image.open('./cat.jpeg')
    jpg = jpg.resize((1080, 1920), Image.BILINEAR)
    jpg.save('./cat1.jpeg')
    if len(sys.argv) < 3:
        print('it needs jpg_path, pgm_path inputs')
    # jpg2pgm(sys.argv[1], sys.argv[2])
    for jpg_file in glob.glob('./*.jpg'):
        print('jpg_file: ', jpg_file)
        jpg2pgm(jpg_file, './')
