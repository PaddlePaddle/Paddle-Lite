# coding=utf-8
import cv2
from array import array


def resize_take_rgbs(path, shape_h_w):
    print '--------------resize_take_rgbs-----------------begin'
    image = cv2.imread(path)
    # print image.shape
    cv2.imshow("before", image)

    print_rgb(image[0, 0])
    # image len may be for .just check it
    # image.resize(shape_h_w)

    image = cv2.resize(image, (shape_h_w[0], shape_h_w[1]))

    cv2.imshow("after", image)
    print image.shape
    height = shape_h_w[0]
    width = shape_h_w[1]

    rs_ = []
    gs_ = []
    bs_ = []
    for h in range(0, height):
        for w in range(0, width):
            bs_.append(image[h, w, 0])
            gs_.append(image[h, w, 1])
            rs_.append(image[h, w, 2])

    # print image[2, 2, 0]/255.
    print len(bs_)
    print len(gs_)
    print len(rs_)
    print '--------------resize_take_rgbs-----------------end'
    return bs_, gs_, rs_


def print_rgb((b, g, r)):
    print "像素 - R:%d,G:%d,B:%d" % (r, g, b)  # 显示像素值
    #
    # image[0, 0] = (100, 150, 200)  # 更改位置(0,0)处的像素
    #
    # (b, g, r) = image[0, 0]  # 再次读取(0,0)像素
    # print "位置(0,0)处的像素 - 红:%d,绿:%d,蓝:%d" % (r, g, b)  # 显示更改后的像素值
    #
    # corner = image[0:100, 0:100]  # 读取像素块
    # cv2.imshow("Corner", corner)  # 显示读取的像素块
    #
    # image[0:100, 0:100] = (0, 255, 0);  # 更改读取的像素块
    #
    # cv2.imshow("Updated", image)  # 显示图像
    #
    # cv2.waitKey(0)  # 程序暂停


def save_to_file(to_file_name, array):
    to_file = open(to_file_name, "wb")
    array.tofile(to_file)
    to_file.close()
