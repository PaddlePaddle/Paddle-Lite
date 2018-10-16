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


data = np.fromfile('datas/img.res')
print data.size
print data[0]

data.reshape(1, 3, 416, 416)
out_array = array('f')
print'--------------------'
print data.size
print data[0]

print '如果是nhwc --------'
# rgb rgb rgb rgb rgb
print data[416 * 3 * 2 + 3 * 2 + 2]
# print data[2]

print '如果是nchw --------'
# rgb rgb rgb rgb rgb
print data[416 * 416 * 2 + 416 * 2 + 2]
# print data[2]

# 明明是nchw

for i in range(0, data.size):
    out_array.append(data[i])

print len(out_array)

print out_array[416 * 416 * 2 + 416 * 2 + 2]

tools.save_to_file('datas/in_put_1_3_416_416_2', out_array)
