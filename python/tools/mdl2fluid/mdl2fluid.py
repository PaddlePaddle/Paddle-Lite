import json
import framework_pb2 as framework_pb2
import op_types as types


def load_mdl(mdl_json_path):
    # print('mdl json path : ' + mdl_json_path)
    with open(mdl_json_path, 'r') as f:
        return json.load(f)


class Converter:
    'convert mdlmodel to fluidmodel'

    def __init__(self, mdl_json_path):
        self.mdl_json_path = mdl_json_path
        print mdl_json_path
        self.mdl_json = load_mdl(self.mdl_json_path)
        self.program_desc = framework_pb2.ProgramDesc()

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
        print 'blocks: '
        print self.program_desc.blocks

    def package_ops(self, block_desc):
        # add ops with layer
        if 'layer' in self.mdl_json:

            layers_ = self.mdl_json['layer']
            for layer in layers_:
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

        if 'param' in layer:
            l_params = layer['param']

            attrs_add = desc_ops_add.attrs.add()
            attrs_add.name = 'paddings'
            # ints
            attrs_add.type = 6
            attrs_add.ints.append(l_params[types.fusion_conv_add_attrs_dict.get('paddings')])
            attrs_add.ints.append(l_params[types.fusion_conv_add_attrs_dict.get('paddings')])

            attrs_add = desc_ops_add.attrs.add()
            attrs_add.name = 'strides'
            # ints
            attrs_add.type = 6
            attrs_add.ints.append(l_params[types.fusion_conv_add_attrs_dict.get('strides')])
            attrs_add.ints.append(l_params[types.fusion_conv_add_attrs_dict.get('strides')])

            attrs_add = desc_ops_add.attrs.add()
            attrs_add.name = 'groups'
            # int
            attrs_add.type = 0
            attrs_add.i = l_params[types.fusion_conv_add_attrs_dict.get('groups')]

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
            outputs_add.parameter = types.op_io_dict.get(desc_ops_add.type).get(types.mdl_outputs_key)
            outputs_add.arguments.append(o)

    @staticmethod
    def package_ops_weight2inputs(desc_ops_add, layer):
        l_weights = layer['weight']
        op_weight_tup = types.op_io_dict.get(desc_ops_add.type).get(types.mdl_weight_key)
        # print len(op_weight_tup)
        for i, val in enumerate(op_weight_tup):
            # print i
            # print val
            inputs_add = desc_ops_add.inputs.add()
            # print w
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


# print mdl_path
# # model
# mdl_model = load_mdl(mdl_path)
# for key in mdl_model:
#     print key
#
# # layer
# layers = mdl_model['layer']
# print layers
#
# for layer in layers:
#     print layer
#     for i in layer:
#         print i
#     if 'name' in layer:
#         l_name = layer['name']
#
#     if 'weight' in layer:
#         l_weights = layer['weight']
#
#     if 'param' in layer:
#         l_params = layer['param']
#
#     if 'output' in layer:
#         l_outputs = layer['output']
#
#     if 'input' in layer:
#         l_inputs = layer['input']
#
#     if 'type' in layer:
#         l_type = layer['type']
#
# print mdl_model['matrix']
#
# package()

mdl_path = "/Users/xiebaiyuan/PaddleProject/paddle-mobile/python/tools/mdl2fluid/multiobjects/YOLO_Universal.json"
converter = Converter(mdl_path)
converter.convert()
