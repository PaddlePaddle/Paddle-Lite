import binascii
import os
import numpy as np


def read_param(path):
    try:
        with open(path, "r") as f:
            value = f.read(2)
            a_hex = binascii.b2a_hex(value)
            print a_hex


            # value = f.read(2)
            # a_hex = binascii.b2a_hex(value)
            # print a_hex
            # value = f.read(2)
            # a_hex = binascii.b2a_hex(value)
            # print a_hex

    except IOError:
        print ": File not found."


def get_file_size(file_path):
    file_path = unicode(file_path, 'utf8')
    f_size = os.path.getsize(file_path)
    f_size = f_size / float(1024 * 1024)
    return round(f_size, 2)


read_param(
    "/Users/xiebaiyuan/PaddleProject/paddle-mobile/python/tools/mdl2fluid/multiobjects/YOLOParameters_Universal"
    ".bundle/conv1_0.bin")
