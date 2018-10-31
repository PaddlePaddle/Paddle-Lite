# coding=utf-8
import json
import os

from core import framework_pb2 as framework_pb2, op_types as types
from mobilenet.swicher import Swichter
import shutil


def load_mdl(mdl_json_path):
    # print('mdl json path : ' + mdl_json_path)
    with open(mdl_json_path, 'r') as f:
        return json.load(f)


def create_if_not_exit(target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, 0777)


class Converter:
    'convert mdlmodel to fluidmodel'

    def __init__(self, base_dir, mdl_json_path):
        print 'base_dir:  ' + base_dir
        self.mdl_json_path = base_dir + mdl_json_path
        self.base_dir = base_dir
        print mdl_json_path
        self.source_weights_dir = self.base_dir + 'datas/sourcemodels/source_weights/'
        self.target_weight_dir = self.base_dir + 'datas/target/target_weights/'

        create_if_not_exit(self.target_weight_dir)

        self.mdl_json = load_mdl(self.mdl_json_path)
        self.program_desc = framework_pb2.ProgramDesc()
        self.weight_list_ = []
        self.deepwise_weight_list_ = []
        # print(json_dick)
        # layers = (json_dick['layer'])
        # for layer in layers:
        #     print(layer)

    def convert(self):
        print 'convert begin.....'
        # add block_desc
        block_desc = self.program_desc.blocks.add()
        block_desc.idx = 0
        block_desc.parent_idx = -1
        self.package_ops(block_desc)
        self.package_vars(block_desc)
        print 'blocks: '
        print self.program_desc.blocks
        print 'convert end.....'
        desc_serialize_to_string = self.program_desc.SerializeToString()

        outputmodel_dir = self.base_dir + 'datas/target/mobilenet_classfication/'
        if os.path.exists(outputmodel_dir):
            shutil.rmtree(outputmodel_dir)
        os.makedirs(outputmodel_dir, 0777)

        if os.path.exists(outputmodel_dir):
            shutil.rmtree(outputmodel_dir)
        # create_if_not_exit(outputmodel_dir)

        shutil.copytree(self.target_weight_dir, outputmodel_dir)

        f = open(outputmodel_dir + "__model__", "wb")
        f.write(desc_serialize_to_string)
        f.close()

    def package_ops(self, block_desc):

        self.add_op_feed(block_desc)

        # add ops with layer
        if 'layer' in self.mdl_json:

            layers_ = self.mdl_json['layer']
            for layer in layers_:

                if layer['type'] == 'SoftmaxLayer':
                    pass
                else:
                    desc_ops_add = block_desc.ops.add()

                    # print layer
                    # for i in layer:
                    #     print i
                    if 'name' in layer:
                        l_name = layer['name']
                    if 'type' in layer:
                        self.package_ops_type(desc_ops_add, layer)

                    if 'weight' in layer:
                        self.package_ops_weight2inputs(desc_ops_add, layer)

                    if 'output' in layer:
                        self.package_ops_outputs(desc_ops_add, layer)

                    if 'input' in layer:
                        self.package_ops_inputs(desc_ops_add, layer)

                    self.package_ops_attrs(desc_ops_add, layer)

        self.add_op_fetch(block_desc)

    def add_op_feed(self, block_desc):
        desc_ops_add = block_desc.ops.add()
        inputs_add = desc_ops_add.inputs.add()
        inputs_add.parameter = 'X'
        inputs_add.arguments.append('feed')
        desc_ops_add.type = 'feed'
        outputs_add = desc_ops_add.outputs.add()
        outputs_add.parameter = 'Out'
        outputs_add.arguments.append('data')
        attrs_add = desc_ops_add.attrs.add()
        attrs_add.name = 'col'
        # boolean
        attrs_add.type = 0
        attrs_add.i = 0

    def add_op_fetch(self, block_desc):
        desc_ops_add = block_desc.ops.add()
        inputs_add = desc_ops_add.inputs.add()
        inputs_add.parameter = 'X'
        # todo pick last layer --> op output
        inputs_add.arguments.append('fc7')
        desc_ops_add.type = 'fetch'
        outputs_add = desc_ops_add.outputs.add()
        outputs_add.parameter = 'Out'
        outputs_add.arguments.append('fetch')
        attrs_add = desc_ops_add.attrs.add()
        attrs_add.name = 'col'
        # boolean
        attrs_add.type = 0
        attrs_add.i = 0

    @staticmethod
    def package_ops_attrs(desc_ops_add, layer):
        # print l_params
        # print desc_ops_add.type
        if desc_ops_add.type == types.op_fluid_fusion_conv_add:
            Converter.pack_fusion_conv_add_attr(desc_ops_add, layer)
        elif desc_ops_add.type == types.op_fluid_relu:
            # fusion_conv_add : attrs
            attrs_add = desc_ops_add.attrs.add()
            attrs_add.name = 'use_mkldnn'
            # boolean
            attrs_add.type = 6
            attrs_add.b = 0
        elif desc_ops_add.type == types.op_fluid_pooling:
            Converter.pack_pooling_attr(desc_ops_add, layer)
            pass
        elif desc_ops_add.type == types.op_fluid_softmax:
            pass

    @staticmethod
    def pack_pooling_attr(desc_ops_add, layer):
        print layer
        l_params = layer['param']

        attrs_add = desc_ops_add.attrs.add()
        attrs_add.name = 'use_mkldnn'
        # boolean
        attrs_add.type = 6
        attrs_add.b = 0

        attrs_add = desc_ops_add.attrs.add()
        attrs_add.name = 'use_cudnn'
        # boolean
        attrs_add.type = 6
        attrs_add.b = 1

        attrs_add = desc_ops_add.attrs.add()
        attrs_add.name = 'paddings'
        # ints
        attrs_add.type = 3
        attrs_add.ints.append(0)
        attrs_add.ints.append(0)

        attrs_add = desc_ops_add.attrs.add()
        attrs_add.name = 'strides'
        # ints
        attrs_add.type = 3
        attrs_add.ints.append(1)
        attrs_add.ints.append(1)

        attrs_add = desc_ops_add.attrs.add()
        attrs_add.name = 'global_pooling'
        # boolean
        attrs_add.type = 6
        attrs_add.b = (l_params[types.pool2d_attrs_dict.get('global_pooling')])

        attrs_add = desc_ops_add.attrs.add()
        attrs_add.name = 'pooling_type'
        # 2-->STRING
        attrs_add.type = 2
        # 注意这里 avg but mdl is ave
        attrs_add.s = l_params[types.pool2d_attrs_dict.get('pooling_type')]

        attrs_add = desc_ops_add.attrs.add()
        attrs_add.name = 'ceil_mode'
        # boolean
        attrs_add.type = 6
        attrs_add.b = 1

        attrs_add = desc_ops_add.attrs.add()
        attrs_add.name = 'ksize'
        # ints
        attrs_add.type = 3
        attrs_add.ints.append(7)
        attrs_add.ints.append(7)

    # type: "pool2d"
    # attrs
    # {
    #     name: "use_mkldnn"
    #     type: BOOLEAN
    #     b: false
    # }
    # attrs
    # {
    #     name: "ceil_mode"
    #     type: BOOLEAN
    #     b: true
    # }
    # attrs
    # {
    #     name: "use_cudnn"
    #     type: BOOLEAN
    #     b: true
    # }
    # attrs
    # {
    #     name: "paddings"
    #     type: INTS
    #     ints: 0
    #     ints: 0
    # }
    # attrs
    # {
    #     name: "strides"
    #     type: INTS
    #     ints: 1
    #     ints: 1
    # }
    # attrs
    # {
    #     name: "global_pooling"
    #     type: BOOLEAN
    #     b: false
    # }
    # attrs
    # {
    #     name: "data_format"
    #     type: STRING
    #     s: "AnyLayout"
    # }
    # attrs
    # {
    #     name: "ksize"
    #     type: INTS
    #     ints: 7
    #     ints: 7
    # }
    # attrs
    # {
    #     name: "pooling_type"
    #     type: STRING
    #     s: "avg"
    # }
    # is_target: false

    @staticmethod
    def pack_fusion_conv_add_attr(desc_ops_add, layer):

        # fusion_conv_add : attrs
        attrs_add = desc_ops_add.attrs.add()
        attrs_add.name = 'workspace_size_MB'
        # 0-->INT
        attrs_add.type = 0
        attrs_add.i = 4096

        attrs_add = desc_ops_add.attrs.add()
        attrs_add.name = 'data_format'
        # 2-->STRING
        attrs_add.type = 2
        attrs_add.s = 'AnyLayout'

        attrs_add = desc_ops_add.attrs.add()
        attrs_add.name = 'use_mkldnn'
        # boolean
        attrs_add.type = 6
        attrs_add.b = 0

        attrs_add = desc_ops_add.attrs.add()
        attrs_add.name = 'use_cudnn'
        # boolean
        attrs_add.type = 6
        attrs_add.b = 1

        attrs_add = desc_ops_add.attrs.add()
        attrs_add.name = 'dilations'
        # ints
        attrs_add.type = 3
        attrs_add.ints.append(1)
        attrs_add.ints.append(1)

        attrs_add = desc_ops_add.attrs.add()
        attrs_add.name = 'axis'
        # int
        attrs_add.type = 0
        attrs_add.i = 1

        if 'param' in layer:
            l_params = layer['param']

            attrs_add = desc_ops_add.attrs.add()
            attrs_add.name = 'paddings'
            # ints
            attrs_add.type = 3
            attrs_add.ints.append(l_params[types.fusion_conv_add_attrs_dict.get('paddings')])
            attrs_add.ints.append(l_params[types.fusion_conv_add_attrs_dict.get('paddings')])

            # attrs_add = desc_ops_add.attrs.add()
            # attrs_add.name = 'paddings'
            # # ints
            # attrs_add.type = 3
            # attrs_add.ints.append(0)
            # attrs_add.ints.append(0)

            attrs_add = desc_ops_add.attrs.add()
            attrs_add.name = 'strides'
            # ints
            attrs_add.type = 3
            attrs_add.ints.append(l_params[types.fusion_conv_add_attrs_dict.get('strides')])
            attrs_add.ints.append(l_params[types.fusion_conv_add_attrs_dict.get('strides')])

            # attrs_add = desc_ops_add.attrs.add()
            # attrs_add.name = 'strides'
            # # ints
            # attrs_add.type = 3
            # attrs_add.ints.append(6)
            # attrs_add.ints.append(6)

            attrs_add = desc_ops_add.attrs.add()
            attrs_add.name = 'groups'
            # int
            attrs_add.type = 0
            attrs_add.i = l_params[types.fusion_conv_add_attrs_dict.get('groups')]
            # attrs_add.i = 1

        #
        # op_attrs_tupl = types.op_io_dict.get(desc_ops_add.type) \
        #     .get(types.mdl_attrs_key)
        #
        #
        #
        #
        # # group stride padding
        # print '----------------------'
        # for i, val in enumerate(op_attrs_tupl):
        #     attrs_add = desc_ops_add.attrs.add()
        #     attr_name = op_attrs_tupl[i]
        #     print attr_name
        #     attrs_add.name = attr_name
        #     attrs_add.type = types.fluid_attrs_type_dict.get(attr_name)
        #     attrs_add.
        #     print l_params[types.fusion_conv_add_attrs_dict.get(attr_name)]

        # for p in l_params:
        #     attrs_add = desc_ops_add.attrs.add()

    @staticmethod
    def package_ops_inputs(desc_ops_add, layer):
        l_inputs = layer['input']
        for i in l_inputs:
            inputs_add = desc_ops_add.inputs.add()
            # print i
            inputs_add.parameter = types.op_io_dict.get(desc_ops_add.type).get(types.mdl_inputs_key)
            inputs_add.arguments.append(i)

    @staticmethod
    def package_ops_outputs(desc_ops_add, layer):
        l_outputs = layer['output']
        for o in l_outputs:
            # print o
            outputs_add = desc_ops_add.outputs.add()
            dict = types.op_io_dict.get(desc_ops_add.type)
            # print 'desc_ops_add.type:  ' + desc_ops_add.type
            # print dict
            outputs_add.parameter = dict.get(types.mdl_outputs_key)
            outputs_add.arguments.append(o)

    def package_ops_weight2inputs(self, desc_ops_add, layer):
        l_weights = layer['weight']
        for w in l_weights:
            self.weight_list_.append(w)

        if layer['type'] == types.layer_mdl_deepwise_conv:
            # print l_weights[0]
            self.deepwise_weight_list_.append(l_weights[0])

        op_weight_tup = types.op_io_dict.get(desc_ops_add.type).get(types.mdl_weight_key)
        if op_weight_tup is not None:
            # print len(op_weight_tup)
            for i, val in enumerate(op_weight_tup):
                # print i
                # print val
                inputs_add = desc_ops_add.inputs.add()
                inputs_add.parameter = op_weight_tup[i]
                inputs_add.arguments.append(l_weights[i])

        # for w in l_weights:
        #     inputs_add = desc_ops_add.inputs.add()
        #     # print w
        #     inputs_add.parameter = op_weight_tup[0]
        #     inputs_add.arguments.append(w)

    @staticmethod
    def package_ops_type(desc_ops_add, layer):
        l_type = layer['type']
        # print l_type
        # print mdl2fluid_op_layer_dict.get(l_type)
        desc_ops_add.type = types.mdl2fluid_op_layer_dict.get(l_type)

    def package_vars(self, block_desc):
        vars_add = block_desc.vars.add()
        vars_add.name = 'feed'
        vars_add.type.type = 9  # 9 is FEED_MINIBATCH
        vars_add.persistable = 1
        # fetch
        vars_add = block_desc.vars.add()
        vars_add.name = 'fetch'
        vars_add.type.type = 10  # 10 is fetch list
        vars_add.persistable = 1

        json_matrix_ = self.mdl_json['matrix']
        # print json_matrix_
        for j in json_matrix_:
            vars_add = block_desc.vars.add()
            vars_add.name = j
            vars_add.type.type = 7  # 7 is lodtensor
            # print j
            tensor = vars_add.type.lod_tensor.tensor
            tensor.data_type = 5  # 5 is FP32

            # print json_matrix_

            dims_of_matrix = json_matrix_.get(j)
            # dims_size = len(dims_of_matrix)
            # print dims_size

            # if dims_size == 4:
            #     tensor.dims.append(dims_of_matrix[0])  # N
            #     tensor.dims.append(dims_of_matrix[3])  # C
            #     tensor.dims.append(dims_of_matrix[1])  # H
            #     tensor.dims.append(dims_of_matrix[2])  # W
            # else:

            # issues in mdl model filter swich n and c
            if j in self.deepwise_weight_list_ and len(dims_of_matrix) == 4:
                print "deep wise issue fit:  " + j
                tensor.dims.append(dims_of_matrix[1])
                tensor.dims.append(dims_of_matrix[0])
                tensor.dims.append(dims_of_matrix[2])
                tensor.dims.append(dims_of_matrix[3])
                print tensor.dims
            else:
                for dims in dims_of_matrix:
                    # print dims
                    tensor.dims.append(dims)

            if j in self.weight_list_:
                vars_add.persistable = 1
                dims_size = len(dims_of_matrix)
                # print dims_size
                # print 'weight name : ' + j
                Swichter().copy_add_head(
                    self.source_weights_dir + j + '.bin',
                    self.target_weight_dir + j
                )

                # if dims_size == 4:
                #     # convert weight from nhwc to nchw
                #     Swichter().nhwc2nchw_one_slice_add_head(
                #         'yolo/datas/multiobjects/float32s_nhwc/' + j + '.bin',
                #         'yolo/datas/multiobjects/float32s_nchw_with_head/' + j,
                #         'yolo/datas/multiobjects/float32s_nchw/' + j + '.tmp',
                #         dims_of_matrix[0],
                #         dims_of_matrix[1],
                #         dims_of_matrix[2],
                #         dims_of_matrix[3]
                #     )
                # else:
                #     Swichter().copy_add_head(
                #         'yolo/datas/multiobjects/float32s_nhwc/' + j + '.bin',
                #         'yolo/datas/multiobjects/float32s_nchw_with_head/' + j,
                #         'yolo/datas/multiobjects/float32s_nchw/' + j + '.tmp'
                #     )
            else:
                vars_add.persistable = 0


mdl_path = "datas/sourcemodels/source_profile/mobileNetModel.json"
base_dir = "/Users/xiebaiyuan/PaddleProject/paddle-mobile/tools/python/modeltools/mobilenet/"
converter = Converter(base_dir, mdl_path)
converter.convert()
