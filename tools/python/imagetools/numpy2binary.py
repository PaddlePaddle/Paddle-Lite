# coding=utf-8

# 这个脚本是可以将numpy合并到二进制
import cv2
import numpy as np
import imagetools as tools
from array import array

#
# image = cv2.imread(path)
# print image.shape
#
# print_rgb(image[0, 0])
# # image len may be for .just check it
# image.resize(shape_h_w)


data = np.fromfile('/Users/xiebaiyuan/PaddleProject/paddle-mobile/tools/python/imagetools/datas/jpgs2/0000_0.9834-148196_82452-0ad4b83ec6bc0f9c5f28101539267054.jpg_p0_0.126571263346.jpg.input.npfile','f')
print data.size
print data

data.reshape(1, 3, 224, 224)
out_array = array('f')
print'--------------------'
print data.size
print data[0]

print '如果是nhwc --------'
# rgb rgb rgb rgb rgb
print data[224 * 3 * 2 + 3 * 2 + 2]
# print data[2]

print '如果是nchw --------'
# rgb rgb rgb rgb rgb
print data[224 * 224 * 2 + 224 * 2 + 2]
# print data[2]

# 明明是nchw

for i in range(0, data.size):
    out_array.append(data[i])

print len(out_array)

print out_array[224 * 224 * 2 + 224 * 2 + 2]

# print out_array

tools.save_to_file('datas/in_put_1_3_224_224_nchw', out_array)
