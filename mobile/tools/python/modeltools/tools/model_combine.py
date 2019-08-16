# coding=utf-8
import os

path = "mobilenet/"  # 文件夹目录
to_file_path = "mobilenet_combine/params"
files = os.listdir(path)  # 得到文件夹下的所有文件名称
files.sort(cmp=None, key=str.lower)
to_file = open(to_file_path, "wb")

for file in files:  # 遍历文件夹
    if not os.path.isdir(file) and file != ".DS_Store":  # 判断是否是文件夹，不是文件夹才打开
        f = open(path + "/" + file)  # 打开文件
        name = f.name
        print 'name:  ' + name
        from_file = open(name, "rb")
        to_file.write(from_file.read())
        from_file.close()

to_file.close()
