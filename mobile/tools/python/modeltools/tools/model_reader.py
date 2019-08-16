import os

from core import framework_pb2 as framework_pb2


def read_model(model_path):
    print('read_model.')
    path_8 = unicode(model_path, 'utf8')

    try:
        with open(path_8, "rb") as f_model:
            print get_file_size(model_path)
            desc = framework_pb2.ProgramDesc()
            desc.ParseFromString(f_model.read())
            print desc
            # print desc.blocks

    except IOError:
        print ": File not found."


def get_file_size(file_path):
    file_path = unicode(file_path, 'utf8')
    fsize = os.path.getsize(file_path)
    fsize = fsize / float(1024 * 1024)
    return round(fsize, 2)


path = '/Users/xiebaiyuan/PaddleProject/paddle-mobile/tools/python/modeltools/mobilenet/datas/sourcemodels/mobilenet_example/mobilenet/__model__'
read_model(path)
