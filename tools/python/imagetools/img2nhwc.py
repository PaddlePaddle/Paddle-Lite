# coding=utf-8
import cv2
from array import array
import imagetools as tools


def combine_bgrs_nhwc(bgrs, means_b_g_r, scale):
    print "scale: %f" % scale
    print means_b_g_r
    # print len(bgrs)
    bs = bgrs[0]
    gs = bgrs[1]
    rs = bgrs[2]
    assert len(bs) == len(gs) == len(rs)
    # print len(bs)
    bgrs_float_array = array('f')
    for i in range(0, len(bs)):
        bgrs_float_array.append((rs[i] - means_b_g_r[2]) * scale)  # r
        bgrs_float_array.append((gs[i] - means_b_g_r[1]) * scale)  # g
        bgrs_float_array.append((bs[i] - means_b_g_r[0]) * scale)  # b

    print len(bgrs_float_array)

    print '------------------'
    print bgrs_float_array[0]
    print bgrs_float_array[999]
    return bgrs_float_array


bgrs = tools.resize_take_rgbs('newyolo_1.jpg', (416, 416, 3))
array = combine_bgrs_nhwc(bgrs, (0, 0, 0), 1.0 / 255)
tools.save_to_file('desktop_1_3_416_416_nhwc_float', array)

cv2.waitKey(0)
