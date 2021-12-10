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

import sys
sys.path.append('../')

from auto_scan_test import AutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st
from functools import partial
import random
import numpy as np

# having diff

class TestFcOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        x86_places = [
                     Place(TargetType.X86, PrecisionType.FP32, DataLayoutType.NCHW),
                     Place(TargetType.Host, PrecisionType.FP32, DataLayoutType.NCHW)
                     ]
        self.enable_testing_on_place(places=x86_places)
        # opencl demo
        opencl_places = [Place(TargetType.OpenCL, PrecisionType.FP16, DataLayoutType.ImageDefault),
                          Place(TargetType.OpenCL, PrecisionType.FP16, DataLayoutType.ImageFolder),
                          Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
                          Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.ImageDefault),
                          Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.ImageFolder),
                          Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),
                          Place(TargetType.Host, PrecisionType.FP32)    
                        ]
        self.enable_testing_on_place(places=opencl_places)

    def is_program_valid(self, program_config: ProgramConfig , predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(st.lists(st.integers(min_value=5, max_value=8), min_size = 4, max_size=4))
        
        in_dtype = draw(st.sampled_from([0, 1, 2]))

        def generate_input_data():
            if (in_dtype == 0):
                return np.random.normal(0.0, 1.0, in_shape).astype(np.float32)
            elif (in_dtype == 1):
                return np.random.randint(1, 500, in_shape).astype(np.int32)
            elif (in_dtype == 2):
                return np.random.randint(1, 500, in_shape).astype(np.int64)
        
        starts_data=draw(st.lists(st.integers(min_value=0, max_value=2), min_size=1, max_size=4))
        ends_data=draw(st.lists(st.integers(min_value=3, max_value=4), min_size=1, max_size=4))
        strides_data=draw(st.lists(st.integers(min_value=1, max_value=1), min_size=1, max_size=4))
        axes_data=draw(st.lists(st.integers(min_value=0, max_value=3), min_size=1, max_size=4))
        # whether this axis for runtime calculations
        infer_flags_data=draw(st.lists(st.integers(min_value=1, max_value=1), min_size=1, max_size=4))
    
        assume(len(starts_data) == len(ends_data))
        assume(len(ends_data) == len(strides_data))
        assume(len(strides_data) == len(axes_data))
        assume(len(axes_data) == len(infer_flags_data))

        inputs = {"Input" : ["input_data"]}
        choose_start = draw(st.sampled_from(["starts", "StartsTensor_data", "StartsTensorList_data1"]))
        choose_end = draw(st.sampled_from(["ends", "EndsTensor_data", "EndsTensorList_data1"]))
        choose_stride = draw(st.sampled_from(["strides", "StridesTensor_data", "StridesTensorList_data1"]))


        # some of below 6 inputs are not supported by lite
        # So not add them to the input of the operator
        def generate_StartsTensor_data():
            if (choose_start == "StartsTensor_data"):
                #inputs["StartsTensor"] = ["StartsTensor_data"]
                return np.array(starts_data).astype(np.int32)
            else:
                return np.random.randint(1, 5, []).astype(np.int32)

        def generate_EndsTensor_data():
            if (choose_end == "EndsTensor_data"):
                #inputs["EndsTensor"] = ["EndsTensor_data"]
                return np.array(ends_data).astype(np.int32)
            else:
                return np.random.randint(1, 5, []).astype(np.int32)

        def generate_StridesTensor_data():
            if (choose_stride == "StridesTensor_data"):
                #inputs["StridesTensor"] = ["StridesTensor_data"]
                return np.array(strides_data).astype(np.int32)
            else:
                return np.random.randint(1, 5, []).astype(np.int32)

        def generate_StartsTensorList_data():
            if (choose_start == "StartsTensorList_data"):
                #inputs["StridesTensorList"] = ["StartsTensorList_data"]
                return np.array(starts_data).astype(np.int32)
            else:
                return np.random.randint(1, 5, []).astype(np.int32)

        def generate_EndsTensorList_data():
            if (choose_end == "EndsTensorList_data"):
                #inputs["EndsTensorList"] = ["EndsTensorList_data"]
                return np.array(ends_data).astype(np.int32)
            else:
                return np.random.randint(1, 5, []).astype(np.int32)
        
        def generate_StridesTensorList_data():
            if (choose_stride == "StridesTensorList_data"):
                #inputs["StridesTensorList"] = ["StridesTensorList_data"]
                return np.array(strides_data).astype(np.int32)
            else:
                return np.random.randint(1, 5, []).astype(np.int32)

        strideslice_op = OpConfig(
            type = "strided_slice",
            inputs = inputs,
            outputs = {"Out": ["output_data"]},
            attrs = {"starts": starts_data,
                    "ends": ends_data,
                    "strides": strides_data,
                    "axes": axes_data,
                    "infer_flags": infer_flags_data,
                    "decrease_axis": []
                     })
        program_config = ProgramConfig(
            ops=[strideslice_op],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input_data)),
                "StartsTensorList_data":TensorConfig(data_gen=partial(generate_EndsTensorList_data)),
                "EndsTensorList_data": TensorConfig(data_gen=partial(generate_StartsTensorList_data)),
                "StridesTensorList_data":  TensorConfig(data_gen=partial(generate_StridesTensorList_data)),
                "StartsTensor_data": TensorConfig(data_gen=partial(generate_EndsTensor_data)),
                "EndsTensor_data": TensorConfig(data_gen=partial(generate_StartsTensor_data)),
                "StridesTensor_data": TensorConfig(data_gen=partial(generate_StridesTensor_data)),
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["strided_slice"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)

if __name__ == "__main__":
    unittest.main(argv=[''])
