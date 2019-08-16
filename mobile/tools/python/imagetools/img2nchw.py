# coding=utf-8
import cv2
from array import array
import imagetools as tools
from enum import Enum


class ChannelType(Enum):
    RGB = 0,
    BGR = 1

def combine_bgrs_nchw(bgrs, means_b_g_r=(103.94, 116.78, 123.68), scale=0.017, channel_type=ChannelType.BGR):
    print("[INFO] ---- combine_bgrs_nchw ---- start")
    print("[INFO] scale:{}".format(scale))
    print("[INFO] mean_b_g_r:{}".format(means_b_g_r))
    #print("[INFO] bgrs:{}".format(bgrs))

    bs = bgrs[0]
    gs = bgrs[1]
    rs = bgrs[2]
    assert len(bs) == len(gs) == len(rs)
    print("[INFO] element size of blue channel = len(bs) = {}".format(len(bs)))

    bgrs_float_array = array('f')
    if channel_type == ChannelType.BGR:
        print('[INFO] bgr format')
        for i in range(0, len(bs)):
            bgrs_float_array.append((bs[i] - means_b_g_r[0]) * scale)  # b
        for i in range(0, len(gs)):
            bgrs_float_array.append((gs[i] - means_b_g_r[1]) * scale)  # g
        for i in range(0, len(rs)):
            bgrs_float_array.append((rs[i] - means_b_g_r[2]) * scale)  # r
    elif channel_type == ChannelType.RGB:
        print('[INFO] rgb format')
        for i in range(0, len(rs)):
            bgrs_float_array.append((rs[i] - means_b_g_r[2]) * scale)  # r
        for i in range(0, len(gs)):
            bgrs_float_array.append((gs[i] - means_b_g_r[1]) * scale)  # g
        for i in range(0, len(bs)):
            bgrs_float_array.append((bs[i] - means_b_g_r[0]) * scale)  # b

    '''
    print("lenI(bgrs_float_array)={}".format(len(bgrs_float_array)))
    print '------------------'
    print bgrs_float_array[0]
    print bgrs_float_array[224 * 224 * 2 + 224 * 2 + 2]
    # for i in range(0, 9):
    #     print'bs %d' % i
    #     print bs[i] / 255.
    print bs[224 * 2 + 2] / 255.
    '''
    print("[INFO] ---- combine_bgrs_nchw ---- end")
    return bgrs_float_array


if __name__ == "__main__":
    # set paras
    #input_image_path = 'banana.jpg'
    #input_image_path = "ocr_detect_512x512.png"
    input_image_path = "ocr_recog_48x512.png"

    reshape_dict = {"n":1, "c":3, "h":48, "w":512}
    output_path = input_image_path.replace(input_image_path[-4:],
                                           "_" + "_".join([str(reshape_dict['n']),
                                                           str(reshape_dict['c']),
                                                           str(reshape_dict['h']),
                                                           str(reshape_dict['w']),
                                                           "nchw",
                                                           "float"],))
    channel_type = ChannelType.BGR
    mean_bgr = (103.94, 116.78, 123.68)
    pixel_scale = 0.017
    #mean_bgr = (0, 0, 0)
    #pixel_scale = 1. / 255

    print("[INFO] input_image_path:{}".format(input_image_path))
    print("[INFO] reshape_dict:{}".format(reshape_dict))
    print("[INFO] output_path:{}".format(output_path))
    print("[INFO] mean_bgr:{}".format(mean_bgr))
    print("[INFO] pixel_scale:{}".format(pixel_scale))

    bgrs = tools.resize_take_rgbs(input_image_path, (reshape_dict['h'],
                                                     reshape_dict['w'],
                                                     reshape_dict['c']))
    array = combine_bgrs_nchw(bgrs, mean_bgr, pixel_scale, channel_type)
    tools.save_to_file(output_path, array)
    print("[INFO] save {} successfully".format(output_path))
    #cv2.waitKey(0)
