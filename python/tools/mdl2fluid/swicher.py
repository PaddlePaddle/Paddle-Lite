from array import array


class Swichter:
    def __init__(self):
        pass

    def nhwc2nchw_one_slice(self, from_file_name, to_file_name, batch, channel, height, width):
        from_file = open(from_file_name, "rb")
        to_file = open(to_file_name, "wb")

        float_array = array("f")
        float_array.fromfile(from_file, width * height * batch * channel)
        float_write_array = array("f")

        for b in range(batch):
            for c in range(channel):
                for h in range(height):
                    for w in range(width):
                        float_value = float_array[b * channel * width * height
                                                  + channel * (h * width + w) + c]

                        float_write_array.append(float_value)

        float_write_array.tofile(to_file)
        from_file.close()
        to_file.close()

    def copy(self, from_file_name, to_file_name):
        from_file = open(from_file_name, "rb")
        to_file = open(to_file_name, "wb")

        to_file.write(from_file.read())
        from_file.close()
        to_file.close()

    def nhwc2nchw_one_slice_add_head(self, from_file_name, to_file_name, tmp_file_name, batch, channel, height, width):
        from_file = open(from_file_name, "rb")
        tmp_file = open(tmp_file_name, "wb+")
        float_array = array("f")
        float_array.fromfile(from_file, width * height * batch * channel)
        float_write_array = array("f")

        for b in range(batch):
            for c in range(channel):
                for h in range(height):
                    for w in range(width):
                        float_value = float_array[b * channel * width * height
                                                  + channel * (h * width + w) + c]

                        float_write_array.append(float_value)

        float_write_array.tofile(tmp_file)
        tmp_file.close()
        from_file.close()

        tmp_file = open(tmp_file_name, "rb")
        to_file = open(to_file_name, "wb")

        tmp = tmp_file.read()
        head = self.read_head('/Users/xiebaiyuan/PaddleProject/paddle-mobile/python/tools/mdl2fluid/yolo/conv1_biases')
        to_file.write(head)
        to_file.write(tmp)
        tmp_file.close()
        to_file.close()

    def read_head(self, head_file):
        from_file = open(head_file, "rb")
        read = from_file.read(20)
        # print read
        from_file.close()
        # print read
        return read

    def copy_add_head(self, from_file_name, to_file_name, tmp_file_name):
        from_file = open(from_file_name, "rb")
        to_file = open(to_file_name, "wb")
        # tmp_file = open(tmp_file_name, "wb")

        head = self.read_head('/Users/xiebaiyuan/PaddleProject/paddle-mobile/python/tools/mdl2fluid/yolo/conv1_biases')
        to_file.write(head)
        to_file.write(from_file.read())
        from_file.close()
        to_file.close()
        pass


# Swichter().nhwc2nchw_one_slice(
#     '/Users/xiebaiyuan/PaddleProject/paddle-mobile/python/tools/mdl2fluid/multiobjects/float32s_nhwc/conv5_6_dw_0.bin',
#     '/Users/xiebaiyuan/PaddleProject/paddle-mobile/python/tools/mdl2fluid/multiobjects/float32s_nchw/conv5_6_dw_0', 1,
#     512, 3, 3)
Swichter().read_head('/Users/xiebaiyuan/PaddleProject/paddle-mobile/python/tools/mdl2fluid/yolo/conv1_biases')
