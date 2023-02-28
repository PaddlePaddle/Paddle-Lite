# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, List, Callable, Dict, Any, Set
import numpy as np
import paddle
import paddleslim
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle import compat as cpt
from paddle.fluid.initializer import NumpyArrayInitializer
from paddle.fluid.framework import convert_np_dtype_to_dtype_

from paddle.fluid.contrib.slim.quantization import QuantizationTransformPass
from paddle.fluid.contrib.slim.quantization import QuantizationFreezePass
from paddle.fluid.framework import IrGraph, IrNode, Operator
from paddle.fluid.executor import global_scope

import os


class TensorConfig:
    '''
    A config builder for a input or a weight.
    '''

    def __init__(self,
                 lod: Optional[List[List[int]]]=None,
                 data_gen: Optional[Callable[..., np.array]]=None,
                 shape: Optional[List[List[int]]]=None):
        '''
        shape: The shape of the tensor.
        dtype: The data type of the tensor.
        data: The value of WeightVar. for input, it should be None
        '''
        self.lod = lod
        if data_gen is not None:
            self.data_gen = data_gen
            self.data = data_gen()
            self.dtype = data_gen().dtype
            self.shape = data_gen().shape
        else:
            assert shape is not None, "While data_gen is not defined, shape must not be None"
            self.data = np.random.normal(0.0, 1.0, shape).astype(np.float32)
            self.shape = shape
            self.dtype = self.data.dtype

    def __repr__(self):
        return str({'shape': self.shape, 'lod': self.lod, 'dtype': self.dtype})


class OpConfig:
    '''  A config builder for generating a Op.  '''

    def __init__(self,
                 type: str,
                 inputs: Dict[str, List[str]],
                 outputs: Dict[str, List[str]],
                 attrs: Dict[str, Any]=None,
                 outputs_dtype: Dict[str, np.dtype]=None,
                 **kwargs):
        self.type = type
        self.inputs = inputs
        self.outputs = outputs
        self.outputs_dtype = outputs_dtype
        self.attrs = attrs
        if self.attrs is None:
            self.attrs = dict()
        self.attrs.update(kwargs)

    def __repr__(self):
        log_str = self.type
        log_str += str(self.attrs)
        return log_str


class ProgramConfig:
    '''  A config builder for generating a Program.  '''

    def __init__(self,
                 ops: List[OpConfig],
                 weights: Dict[str, TensorConfig],
                 inputs: Dict[str, TensorConfig],
                 outputs: List[str]):
        self.ops = ops
        # if no weight need to save, we create a place_holder to help seriazlie params.
        if not weights:

            def generate_weight():
                return np.array([1]).astype(np.float32)

            self.weights = {
                "place_holder_weight": TensorConfig(data_gen=generate_weight)
            }
        else:
            self.weights = weights
        self.inputs = inputs
        self.outputs = outputs

    def __repr__(self):
        log_str = ''
        for i in range(len(self.ops)):
            if i != len(self.ops) - 1:
                log_str += repr(self.ops[i]) + ' + '
            else:
                log_str += repr(self.ops[i])
        log_str += ' -- '
        for t, v in self.inputs.items():
            log_str += '[' + t + ': ' + str(v) + ']'
        for t, v in self.weights.items():
            log_str += '[' + t + ': ' + str(v) + ']'

        return log_str


def create_fake_model(program_config):
    '''  Create a Paddle model(in memory) according to the given config.  '''
    paddle.enable_static()
    main_program_desc = core.ProgramDesc()
    util_program = fluid.Program()
    main_block_desc = main_program_desc.block(0)

    var_desc = main_block_desc.var(cpt.to_bytes("feed"))
    var_desc.set_type(core.VarDesc.VarType.FEED_MINIBATCH)
    var_desc.set_persistable(True)

    index = 0
    for name, tensor_config in program_config.inputs.items():
        var_desc = main_block_desc.var(cpt.to_bytes(name))
        var_desc.set_type(core.VarDesc.VarType.LOD_TENSOR)
        var_desc.set_dtype(convert_np_dtype_to_dtype_(tensor_config.dtype))
        var_desc.set_shape(tensor_config.shape)
        var_desc.set_need_check_feed(True)
        if tensor_config.lod is not None:
            var_desc.set_lod_level(len(tensor_config.lod))
        op_desc = main_block_desc._prepend_op()
        op_desc.set_type("feed")
        op_desc.set_input('X', ["feed"])
        op_desc.set_output('Out', [name])
        op_desc._set_attr("col", index)
        index = index + 1

    save_var_map = {}
    for name, tensor_config in program_config.weights.items():
        var_desc = main_block_desc.var(cpt.to_bytes(name))
        var_desc.set_type(core.VarDesc.VarType.LOD_TENSOR)
        var_desc.set_dtype(convert_np_dtype_to_dtype_(tensor_config.dtype))
        var_desc.set_shape(tensor_config.shape)
        var_desc.set_persistable(True)

        save_var_map[name] = util_program.global_block().create_parameter(
            dtype=tensor_config.dtype,
            shape=tensor_config.shape,
            type=core.VarDesc.VarType.LOD_TENSOR,
            name=name,
            initializer=NumpyArrayInitializer(tensor_config.data))
    in_vars = []
    for name in sorted(save_var_map.keys()):
        in_vars.append(save_var_map[name])

    out_var = util_program.global_block().create_var(
        type=core.VarDesc.VarType.RAW, name="out_var_0")
    out_var.desc.set_persistable(True)
    util_program.global_block().append_op(
        type='save_combine',
        inputs={'X': in_vars},
        outputs={'Y': out_var},
        attrs={'file_path': '',
               'save_to_memory': True})
    for op_config in program_config.ops:
        op_desc = main_block_desc.append_op()
        op_desc.set_type(op_config.type)
        for name, values in op_config.inputs.items():
            op_desc.set_input(name, values)
        for name, values in op_config.attrs.items():
            op_desc._set_attr(name, values)
        for name, values in op_config.outputs.items():
            op_desc.set_output(name, values)
            for v in values:
                var_desc = main_block_desc.var(cpt.to_bytes(v))
                var_desc.set_type(core.VarDesc.VarType.LOD_TENSOR)
                var_desc.set_dtype(convert_np_dtype_to_dtype_(np.float32))
                if op_config.outputs_dtype is not None and v in op_config.outputs_dtype.keys(
                ):
                    var_desc.set_dtype(
                        convert_np_dtype_to_dtype_(op_config.outputs_dtype[v]))

        op_desc.infer_var_type(main_block_desc)
        op_desc.infer_shape(main_block_desc)
        op_desc.check_attrs()

    for index, name in enumerate(program_config.outputs):
        var_desc = main_block_desc.var(cpt.to_bytes("fetch"))
        var_desc.set_type(core.VarDesc.VarType.FETCH_LIST)
        var_desc.set_need_check_feed(True)
        op_desc = main_block_desc.append_op()
        op_desc.set_type("fetch")
        op_desc.set_input('X', [name])
        op_desc.set_output('Out', ["fetch"])
        op_desc._set_attr("col", index)

    main_program_desc._set_version()
    paddle.fluid.core.save_op_version_info(main_program_desc)

    model = main_program_desc.serialize_to_string()

    util_program._sync_with_cpp()
    place = fluid.CPUPlace()
    executor = fluid.Executor(place)
    scope = fluid.Scope()
    with fluid.scope_guard(scope):
        executor.run(util_program)
        params = scope.find_var("out_var_0").get_bytes()
    return model, params


def create_quant_model(model, params, prefix, program_config):
    # 1. store original model
    with open(prefix + "/model", "wb") as f:
        f.write(model)
    with open(prefix + "/params", "wb") as f:
        f.write(params)

    # 2. define calibration data
    paddle.enable_static()
    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)

    batch_size = 1

    def _reader_list():
        for _ in range(10):
            res = []
            for key in program_config.inputs.keys():
                input_shape = program_config.inputs[key].shape
                batch_size = input_shape[0]
                res.append(np.random.random(input_shape).astype(np.float32))
            yield res

    # 3. quant_post_static
    quantize_model_path = prefix + "/static_quantized_conv_2d"
    paddleslim.quant.quant_post_static(
        executor=exe,
        weight_bits=8,
        batch_size=batch_size,
        model_dir=prefix,
        quantize_model_path=quantize_model_path,
        sample_generator=_reader_list,
        weight_quantize_type='abs_max',
        quantizable_op_type=[
            "conv2d", "depthwise_conv2d", "conv2d_transpose", "mul", "matmul"
        ],
        model_filename="model",
        params_filename="params", )

    # 4. return quant model
    with open(quantize_model_path + "/__model__", "rb") as f:
        model = f.read()
    with open(quantize_model_path + "/__params__", "rb") as f:
        params = f.read()
    return model, params


from typing import Optional
from enum import Enum


class TargetType(Enum):
    Unk = 0
    Host = 1
    X86 = 2
    CUDA = 3
    ARM = 4
    OpenCL = 5
    Any = 6
    FPGA = 7
    NPU = 8
    XPU = 9
    BM = 10
    MLU = 11
    RKNPU = 12
    APU = 13
    HUAWEI_ASCEND_NPU = 14
    IMAGINATION_NNA = 15
    INTEL_FPGA = 16
    Metal = 17
    NNAdapter = 18


class PrecisionType(Enum):
    Unk = 0
    FP32 = 1
    INT8 = 2
    INT32 = 3
    Any = 4
    FP16 = 5
    BOOL = 6
    INT64 = 7
    INT16 = 8
    UINT8 = 9
    FP64 = 10


class DataLayoutType(Enum):
    Unk = 0
    NCHW = 1
    Any = 2
    NHWC = 3
    ImageDefault = 4
    ImageFolder = 5
    ImageNW = 6
    MetalTexture2DArray = 7
    MetalTexture2D = 8


def Place(target_type: TargetType,
          precision_type: Optional[PrecisionType]=None,
          data_layout: Optional[DataLayoutType]=None):
    place = target_type.name
    if precision_type != None:
        place = place + "," + precision_type.name
        if data_layout != None:
            place = place + "," + data_layout.name
    return place


class CxxConfig:
    def __init__(self):
        self.config = {}
        self.config["discarded_passes"] = []

    def set_valid_places(self, places):
        self.config["valid_targets"] = places

    def set_threads(self, thread):
        self.config["thread"] = thread

    def set_power_mode(self, mode):
        self.config["power_mode"] = mode

    def add_discarded_pass(self, discarded_pass):
        self.config["discarded_passes"].append(discarded_pass)

    def value(self):
        return self.config

    def target(self):
        if not "valid_targets" in self.config:
            return None
        first_place = self.config["valid_targets"][0].split(",")
        return eval("TargetType." + first_place[0])

    def precision(self):
        if not "valid_targets" in self.config:
            return None
        first_place = ''.join(self.config["valid_targets"][0]).split(",")
        if len(first_place) < 2:
            return PrecisionType.FP32
        else:
            return eval("PrecisionType." + first_place[1])

    def layout(self):
        if not "valid_targets" in self.config:
            return None
        first_place = ''.join(self.config["valid_targets"][0]).split(",")
        if len(first_place) < 3:
            return DataLayoutType.NCHW
        else:
            return eval("DataLayoutType." + first_place[2])

    def set_nnadapter_device_names(self, nnadapter_device_names):
        self.config['nnadapter_device_names'] = nnadapter_device_names

    def set_nnadapter_context_properties(self, nnadapter_context_properties):
        self.config[
            'nnadapter_context_properties'] = nnadapter_context_properties

    def set_nnadapter_model_cache_dir(self, nnadapter_model_cache_dir):
        self.config['nnadapter_model_cache_dir'] = nnadapter_model_cache_dir

    def set_nnadapter_subgraph_partition_config_path(
            self, nnadapter_subgraph_partition_config_path):
        self.config[
            'nnadapter_subgraph_partition_config_path'] = nnadapter_subgraph_partition_config_path

    def set_nnadapter_mixed_precision_quantization_config_path(
            self, nnadapter_mixed_precision_quantization_config_path):
        self.config[
            'nnadapter_mixed_precision_quantization_config_path'] = nnadapter_mixed_precision_quantization_config_path
