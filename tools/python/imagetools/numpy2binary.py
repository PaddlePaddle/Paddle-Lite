#!/usr/bin/env bash
# coding=utf-8

# This script convert numpy format to binary's
import cv2
import numpy as np
import imagetools as tools
from array import array


'''
image = cv2.imread(path)
print image.shape
print_rgb(image[0, 0])
# mage len may be for .just check it
image.resize(shape_h_w)
'''

if __name__ == "__main__":
    # input params
    reshape_dict = {"n": 1, "c": 3, "h": 224, "w": 224}
    np_file_path = 'banana_1_3_224_224_nchw_float'
    save_file_name = 'in_put_1_3_224_224_nchw'

    # load input etc.
    np = np.fromfile(np_file_path, 'f')
    #np = cv2.imread(np_file_path)
    print("np.size:{}".format(np.size))
    print("np:{}".format(np))
    np.reshape(reshape_dict['n'],
               reshape_dict['c'],
               reshape_dict['h'],
               reshape_dict['w'])
    out_array = array('f')

    '''
    print("--------------------")
    print("np.size:{}".format(np.size))
    print("np[0]:{}".format(np[0])

    print("如果是nhw")
    # rgb rgb rgb rgb rgb
    print np[224 * 3 * 2 + 3 * 2 + 2]
    # print np[2]

    print '如果是nchw --------'
    # rgb rgb rgb rgb rgb
    print(np[224 * 224 * 2 + 224 * 2 + 2])
    # print np[2]
    # 明明是nchw
    '''

    for i in range(0, np.size):
        out_array.append(np[i])

    print("len(out_array):{}".format(len(out_array)))
    print("out_array[224 * 224 * 2 + 224 * 2 + 2]:{}".format(out_array[224 * 224 * 2 + 224 * 2 + 2]))

    # print out_array
    tools.save_to_file(save_file_name, out_array)
