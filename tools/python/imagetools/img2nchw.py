# coding=utf-8
import cv2
from array import array
import imagetools as tools
from enum import Enum


class ChannelType(Enum):
    RGB = 0,
    BGR = 1


def combine_bgrs_nchw(bgrs, means_b_g_r, scale, channel_type=ChannelType.BGR):
    print '--------------combine_bgrs_nchw-----------------begin'
    print "scale: %f" % scale
    print means_b_g_r
    # print len(bgrs)
    bs = bgrs[0]
    gs = bgrs[1]
    rs = bgrs[2]

    assert len(bs) == len(gs) == len(rs)
    print len(bs)
    bgrs_float_array = array('f')

    if channel_type == ChannelType.BGR:
        print 'bgr'
        for i in range(0, len(bs)):
            bgrs_float_array.append((bs[i] - means_b_g_r[0]) * scale)  # b
        for i in range(0, len(gs)):
            bgrs_float_array.append((gs[i] - means_b_g_r[1]) * scale)  # g
        for i in range(0, len(rs)):
            bgrs_float_array.append((rs[i] - means_b_g_r[2]) * scale)  # r
    elif channel_type == ChannelType.RGB:
        print 'rgb'

        for i in range(0, len(rs)):
            bgrs_float_array.append((rs[i] - means_b_g_r[2]) * scale)  # r
        for i in range(0, len(gs)):
            bgrs_float_array.append((gs[i] - means_b_g_r[1]) * scale)  # g
        for i in range(0, len(bs)):
            bgrs_float_array.append((bs[i] - means_b_g_r[0]) * scale)  # b

    print len(bgrs_float_array)

    print '------------------'
    print bgrs_float_array[0]
    print bgrs_float_array[224 * 224 * 2 + 224 * 2 + 2]

    # for i in range(0, 9):
    #     print'bs %d' % i
    #     print bs[i] / 255.

    print bs[224 * 2 + 2] / 255.
    print '--------------combine_bgrs_nchw-----------------end'

    return bgrs_float_array


# bgrs = tools.resize_take_rgbs('banana.jpeg', (224, 224, 3))
# array = combine_bgrs_nchw(bgrs, (103.94, 116.78, 123.68), 0.017, array,ChannelType.BGR)
# tools.save_to_file('banana_1_3_224_224_nchw_float')

# cv2.waitKey(0)


bgrs = tools.resize_take_rgbs('datas/jpgs/0000_0.9834-148196_82452-0ad4b83ec6bc0f9c5f28101539267054.jpg_p0_0.126571263346.jpg', (224, 224, 3))
array = combine_bgrs_nchw(bgrs, (0, 0, 0), 1. / 255, ChannelType.RGB)
tools.save_to_file('datas/desktop_1_3_224_224_nchw_float', array)
