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
sys.path.append('..')

from auto_scan_test import FusePassAutoScanTest
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest
import math

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


def prior_box_expand_aspect_ratios(aspect_ratios, flip, output_aspect_ratior):
    epsilon = 1e-6
    output_aspect_ratior.append(1.0)
    for i in range(len(aspect_ratios)):
        ar = aspect_ratios[i]
        already_exist = False
        for j in range(len(output_aspect_ratior)):
            if math.fabs(ar - output_aspect_ratior[j]) < epsilon:
                already_exist = True
                break

        if already_exist == False:
            output_aspect_ratior.append(ar)
            if flip == True:
                output_aspect_ratior.append(1.0 / ar)


def sample_program_configs(draw):
    density_prior_box_or_prior_box_op = draw(
        st.sampled_from(["prior_box", "density_prior_box"]))
    reshape_or_flatten_op_type = draw(
        st.sampled_from(["reshape2", "flatten", "flatten2"]))

    #image params
    batch_size = draw(st.integers(min_value=1, max_value=4))
    image_channels = draw(st.integers(min_value=1, max_value=10))
    image_w = draw(st.integers(min_value=1, max_value=160))
    image_h = draw(st.integers(min_value=1, max_value=160))

    density_prior_box_layer1_channels = draw(
        st.integers(
            min_value=1, max_value=32))
    density_prior_box_layer1_w = draw(st.integers(min_value=1, max_value=40))
    density_prior_box_layer1_h = draw(st.integers(min_value=1, max_value=40))

    density_prior_box_layer2_channels = draw(
        st.integers(
            min_value=1, max_value=64))
    density_prior_box_layer2_w = draw(st.integers(min_value=1, max_value=32))
    density_prior_box_layer2_h = draw(st.integers(min_value=1, max_value=32))

    assume(density_prior_box_layer1_h < image_h)
    assume(density_prior_box_layer2_h < image_h)
    assume(density_prior_box_layer1_w < image_w)
    assume(density_prior_box_layer2_w < image_w)

    def generate_image(*args, **kwargs):
        return np.random.random((batch_size, image_channels, image_h,
                                 image_w)).astype('float32')

    def generate_density_prior_box1(*args, **kwargs):
        return np.random.random((batch_size, density_prior_box_layer1_channels,
                                 density_prior_box_layer1_h,
                                 density_prior_box_layer1_w)).astype('float32')

    def generate_density_prior_box2(*args, **kwargs):
        return np.random.random((batch_size, density_prior_box_layer2_channels,
                                 density_prior_box_layer2_h,
                                 density_prior_box_layer2_w)).astype('float32')

    min_max_aspect_ratios_order = draw(st.sampled_from([True, False]))
    flip = draw(st.sampled_from([True, False]))
    clip = draw(st.sampled_from([True, False]))
    variances = draw(
        st.lists(
            st.floats(
                min_value=0.001, max_value=1), min_size=4, max_size=4))
    value = draw(st.floats(min_value=0, max_value=1))
    step_h_1 = float(image_h) / float(density_prior_box_layer1_h)
    step_w_1 = float(image_w) / float(density_prior_box_layer1_w)
    step_h_2 = float(image_h) / float(density_prior_box_layer2_h)
    step_w_2 = float(image_w) / float(density_prior_box_layer2_w)
    offset_1 = draw(st.floats(min_value=0, max_value=1))
    offset_2 = draw(st.floats(min_value=0, max_value=1))

    if density_prior_box_or_prior_box_op == "prior_box":
        # min_sizes = [2.0, 4.0]
        # max_sizes = [5.0, 10.0]
        # aspect_ratios = [2.0, 3.0]
        min_sizes_int = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=4), min_size=1, max_size=4))
        max_sizes_int = draw(
            st.lists(
                st.integers(
                    min_value=5, max_value=10), min_size=1, max_size=4))
        assume(len(min_sizes_int) == len(max_sizes_int))
        min_sizes = []
        max_sizes = []
        for i in min_sizes_int:
            min_sizes.append(float(i))
        for i in max_sizes_int:
            max_sizes.append(float(i))

        aspect_ratios = draw(
            st.lists(
                st.floats(
                    min_value=0.1, max_value=3), min_size=1, max_size=4))

        #density_prior_box InferShape
        output_aspect_ratior = []
        prior_box_expand_aspect_ratios(
            aspect_ratios, flip, output_aspect_ratior)  #compute output size
        num_priors_1 = len(output_aspect_ratior) * len(
            min_sizes)  #3 need compute
        num_priors_1 += len(max_sizes)
        num_priors_2 = len(output_aspect_ratior) * len(min_sizes)
        num_priors_2 += len(max_sizes)
    else:
        fixed_sizes_1 = draw(
            st.lists(
                st.floats(
                    min_value=2, max_value=32), min_size=3, max_size=3))
        densities_1 = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=2), min_size=3, max_size=3))
        fixed_ratios_1 = draw(
            st.lists(
                st.floats(
                    min_value=1, max_value=1), min_size=1, max_size=1))

        fixed_sizes_2 = draw(
            st.lists(
                st.floats(
                    min_value=2, max_value=32), min_size=2, max_size=2))
        densities_2 = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=2), min_size=2, max_size=2))
        fixed_ratios_2 = draw(
            st.lists(
                st.floats(
                    min_value=1, max_value=1), min_size=1, max_size=1))

        #density_prior_box InferShape
        num_priors_1 = 0
        for index in range(len(densities_1)):
            num_priors_1 += len(fixed_ratios_1) * (pow(densities_1[index], 2))

        num_priors_2 = 0
        for index in range(len(densities_2)):
            num_priors_2 += len(fixed_ratios_2) * (pow(densities_2[index], 2))

    density_prior_box_layer1_length = density_prior_box_layer1_h * density_prior_box_layer1_w * num_priors_1
    density_prior_box_layer2_length = density_prior_box_layer2_h * density_prior_box_layer2_w * num_priors_2
    priorbox_shape = [
        density_prior_box_layer1_length + density_prior_box_layer2_length, 4
    ]

    #reshape2 params
    reshape2_w = draw(st.integers(min_value=4, max_value=4))
    #box_coder params
    code_type = draw(
        st.sampled_from(["decode_center_size", "encode_center_size"]))  #must
    axis = draw(st.sampled_from([0, 1]))  #must 0, else has diff
    box_normalized = True  #box coder opencl kernel required
    variance = draw(st.sampled_from([[0.1, 0.2, 0.3, 0.4], []]))
    lod_data = [[1, 1, 1, 1, 1]]
    assume(priorbox_shape[0] < 16384)

    if code_type == "encode_center_size":
        targetbox_shape = draw(st.sampled_from(
            [[30, 4], [80, 4]]))  #4 is required in paddle
    else:
        num0 = 1
        num1 = 1
        num2 = 1
        if axis == 0:
            num1 = priorbox_shape[0]
            num0 = np.random.randint(1, 100)
        else:
            num0 = priorbox_shape[0]
            num1 = np.random.randint(1, 100)
        num2 = priorbox_shape[1]
        targetbox_shape = draw(st.sampled_from([[num0, num1, num2]]))

    def generate_targetbox(*args, **kwargs):
        return np.random.random(targetbox_shape).astype(np.float32)

    if density_prior_box_or_prior_box_op == "prior_box":
        density_prior_box_or_prior_box_op_1 = OpConfig(
            type="prior_box",
            inputs={
                "Input": ["density_prior_box_input_data_1"],
                "Image": ["density_prior_box_input_image"]
            },
            outputs={
                "Boxes": ["density_prior_box_output_boxes_1"],
                "Variances": ["density_prior_box_output_variances_1"]
            },
            attrs={
                "flip": flip,
                "clip": clip,
                "value": value,
                "min_max_aspect_ratios_order": min_max_aspect_ratios_order,
                "variances": variances,
                "step_h": step_h_1,
                "step_w": step_w_1,
                "offset": offset_1,
                "min_sizes": min_sizes,
                "max_sizes": max_sizes,
                "aspect_ratios": aspect_ratios,
                "flatten_to_2d": False
            })

        density_prior_box_or_prior_box_op_2 = OpConfig(
            type="prior_box",
            inputs={
                "Input": ["density_prior_box_input_data_2"],
                "Image": ["density_prior_box_input_image"]
            },
            outputs={
                "Boxes": ["density_prior_box_output_boxes_2"],
                "Variances": ["density_prior_box_output_variances_2"]
            },
            attrs={
                "flip": flip,
                "clip": clip,
                "value": value,
                "min_max_aspect_ratios_order": min_max_aspect_ratios_order,
                "variances": variances,
                "step_h": step_h_2,
                "step_w": step_w_2,
                "offset": offset_2,
                "min_sizes": min_sizes,
                "max_sizes": max_sizes,
                "aspect_ratios": aspect_ratios,
                "flatten_to_2d": False
            })
    else:
        density_prior_box_or_prior_box_op_1 = OpConfig(
            type="density_prior_box",
            inputs={
                "Input": ["density_prior_box_input_data_1"],
                "Image": ["density_prior_box_input_image"]
            },
            outputs={
                "Boxes": ["density_prior_box_output_boxes_1"],
                "Variances": ["density_prior_box_output_variances_1"]
            },
            attrs={
                "clip": clip,
                "value": value,
                "variances": variances,
                "step_h_1": step_h_1,
                "step_w_1": step_w_1,
                "offset": offset_1,
                "fixed_sizes": fixed_sizes_1,
                "fixed_ratios": fixed_ratios_1,
                "densities": densities_1,
                "flatten_to_2d": False
            })

        density_prior_box_or_prior_box_op_2 = OpConfig(
            type="density_prior_box",
            inputs={
                "Input": ["density_prior_box_input_data_2"],
                "Image": ["density_prior_box_input_image"]
            },
            outputs={
                "Boxes": ["density_prior_box_output_boxes_2"],
                "Variances": ["density_prior_box_output_variances_2"]
            },
            attrs={
                "clip": clip,
                "value": value,
                "variances": variances,
                "step_h_2": step_h_2,
                "step_w_2": step_w_2,
                "offset": offset_2,
                "fixed_sizes": fixed_sizes_2,
                "fixed_ratios": fixed_ratios_2,
                "densities": densities_2,
                "flatten_to_2d": False
            })

    if reshape_or_flatten_op_type == "reshape2":
        reshape_or_flatten_op_1 = OpConfig(
            type="reshape2",
            inputs={"X": ["density_prior_box_output_boxes_1"]},
            outputs={
                "Out": ["reshape_or_flatten_out_1"],
                "XShape": ["reshape2_xshape_1"]
            },
            attrs={"shape": [-1, reshape2_w]})

        reshape_or_flatten_op_2 = OpConfig(
            type="reshape2",
            inputs={"X": ["density_prior_box_output_boxes_2"]},
            outputs={
                "Out": ["reshape_or_flatten_out_2"],
                "XShape": ["reshape2_xshape_2"]
            },
            attrs={"shape": [-1, reshape2_w]})

        reshape_or_flatten_op_3 = OpConfig(
            type="reshape2",
            inputs={"X": ["density_prior_box_output_variances_1"]},
            outputs={
                "Out": ["reshape_or_flatten_out_3"],
                "XShape": ["reshape2_xshape_3"]
            },
            attrs={"shape": [-1, reshape2_w]})

        reshape_or_flatten_op_4 = OpConfig(
            type="reshape2",
            inputs={"X": ["density_prior_box_output_variances_2"]},
            outputs={
                "Out": ["reshape_or_flatten_out_4"],
                "XShape": ["reshape2_xshape_4"]
            },
            attrs={"shape": [-1, reshape2_w]})
    elif reshape_or_flatten_op_type == "flatten":
        reshape_or_flatten_op_1 = OpConfig(
            type="flatten",
            inputs={"X": ["density_prior_box_output_boxes_1"]},
            outputs={"Out": ["reshape_or_flatten_out_1"]},
            attrs={"axis": 3, })

        reshape_or_flatten_op_2 = OpConfig(
            type="flatten",
            inputs={"X": ["density_prior_box_output_boxes_2"]},
            outputs={"Out": ["reshape_or_flatten_out_2"]},
            attrs={"axis": 3, })

        reshape_or_flatten_op_3 = OpConfig(
            type="flatten",
            inputs={"X": ["density_prior_box_output_variances_1"]},
            outputs={"Out": ["reshape_or_flatten_out_3"]},
            attrs={"axis": 3, })

        reshape_or_flatten_op_4 = OpConfig(
            type="flatten",
            inputs={"X": ["density_prior_box_output_variances_2"]},
            outputs={"Out": ["reshape_or_flatten_out_4"]},
            attrs={"axis": 3, })
    else:
        reshape_or_flatten_op_1 = OpConfig(
            type="flatten2",
            inputs={"X": ["density_prior_box_output_boxes_1"]},
            outputs={
                "Out": ["reshape_or_flatten_out_1"],
                "XShape": ["reshape_or_flatten_xshape_1"]
            },
            attrs={"axis": 3, })

        reshape_or_flatten_op_2 = OpConfig(
            type="flatten2",
            inputs={"X": ["density_prior_box_output_boxes_2"]},
            outputs={
                "Out": ["reshape_or_flatten_out_2"],
                "XShape": ["reshape_or_flatten_xshape_2"]
            },
            attrs={"axis": 3, })

        reshape_or_flatten_op_3 = OpConfig(
            type="flatten2",
            inputs={"X": ["density_prior_box_output_variances_1"]},
            outputs={
                "Out": ["reshape_or_flatten_out_3"],
                "XShape": ["reshape_or_flatten_xshape_3"]
            },
            attrs={"axis": 3, })

        reshape_or_flatten_op_4 = OpConfig(
            type="flatten2",
            inputs={"X": ["density_prior_box_output_variances_2"]},
            outputs={
                "Out": ["reshape_or_flatten_out_4"],
                "XShape": ["reshape_or_flatten_xshape_4"]
            },
            attrs={"axis": 3, })

    axis_1 = draw(st.sampled_from([0]))
    axis_2 = draw(st.sampled_from([0]))

    concat_op_1 = OpConfig(
        type="concat",
        inputs={
            "X": ["reshape_or_flatten_out_1", "reshape_or_flatten_out_2"]
        },
        outputs={"Out": ["concat_out_1"]},
        attrs={"axis": axis_1})

    concat_op_2 = OpConfig(
        type="concat",
        inputs={
            "X": ["reshape_or_flatten_out_3", "reshape_or_flatten_out_4"]
        },
        outputs={"Out": ["concat_out_2"]},
        attrs={"axis": axis_2})

    box_coder_op = OpConfig(
        type="box_coder",
        inputs={
            "PriorBox": ["concat_out_1"],
            "TargetBox": ["targetbox_data"],
            "PriorBoxVar": ["concat_out_2"]
        },
        outputs={"OutputBox": ["outputbox_data"]},
        attrs={
            "code_type": code_type,
            "box_normalized": box_normalized,
            "axis": axis,
            "variance": []
        })

    ops = [
        density_prior_box_or_prior_box_op_1,
        density_prior_box_or_prior_box_op_2, reshape_or_flatten_op_1,
        reshape_or_flatten_op_2, reshape_or_flatten_op_3,
        reshape_or_flatten_op_4, concat_op_1, concat_op_2, box_coder_op
    ]
    program_config = ProgramConfig(
        ops=ops,
        weights={},
        inputs={
            "density_prior_box_input_data_1":
            TensorConfig(data_gen=partial(generate_density_prior_box1)),
            "density_prior_box_input_data_2":
            TensorConfig(data_gen=partial(generate_density_prior_box2)),
            "density_prior_box_input_image":
            TensorConfig(data_gen=partial(generate_image)),
            "targetbox_data": TensorConfig(
                data_gen=partial(generate_targetbox), lod=lod_data)
        },
        outputs=["outputbox_data"])
    return program_config


class TestSSDBoxesCalcOfflinePass(FusePassAutoScanTest):
    def __init__(self, *args, **kwargs):
        FusePassAutoScanTest.__init__(self, *args, **kwargs)
        #opencl
        opencl_places = [
            Place(TargetType.OpenCL, PrecisionType.FP16,
                  DataLayoutType.ImageDefault), Place(
                      TargetType.OpenCL, PrecisionType.FP16,
                      DataLayoutType.ImageFolder),
            Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
            Place(TargetType.OpenCL, PrecisionType.Any,
                  DataLayoutType.ImageDefault), Place(
                      TargetType.OpenCL, PrecisionType.Any,
                      DataLayoutType.ImageFolder),
            Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        self.enable_testing_on_place(places=opencl_places)

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        if program_config.ops[8].attrs["axis"] == 1:
            print("The box_coder_op only support axis 0. Skip!")
            return False
        if program_config.ops[8].attrs["code_type"] == "encode_center_size":
            print("The box_coder_op only support decode_center_size. Skip!")
            return False
        return True

    def sample_program_configs(self, draw):
        return sample_program_configs(draw)

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ['box_coder'], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(
            quant=False,
            max_examples=200,
            passes=["ssd_boxes_calc_offline_pass"])


if __name__ == "__main__":
    unittest.main(argv=[''])
