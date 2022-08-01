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
    prior_box_type = draw(st.sampled_from(["density_prior_box"]))
    if prior_box_type == "density_prior_box":
        #image params
        batch_size = draw(st.integers(min_value=1, max_value=4))
        image_channels = 3
        image_w = draw(st.integers(min_value=40, max_value=50))
        image_h = draw(st.integers(min_value=40, max_value=50))

        density_prior_box_layer1_channels = draw(
            st.integers(
                min_value=10, max_value=40))
        density_prior_box_layer1_w = draw(
            st.integers(
                min_value=30, max_value=40))
        density_prior_box_layer1_h = draw(
            st.integers(
                min_value=30, max_value=40))

        density_prior_box_layer2_channels = draw(
            st.integers(
                min_value=10, max_value=40))
        density_prior_box_layer2_w = draw(
            st.integers(
                min_value=30, max_value=40))
        density_prior_box_layer2_h = draw(
            st.integers(
                min_value=30, max_value=40))

        def generate_image(*args, **kwargs):
            return np.random.random((batch_size, image_channels, image_h,
                                     image_w)).astype('float32')

        def generate_density_prior_box1(*args, **kwargs):
            return np.random.random(
                (batch_size, density_prior_box_layer1_channels,
                 density_prior_box_layer1_h,
                 density_prior_box_layer1_w)).astype('float32')

        def generate_density_prior_box2(*args, **kwargs):
            return np.random.random(
                (batch_size, density_prior_box_layer2_channels,
                 density_prior_box_layer2_h,
                 density_prior_box_layer2_w)).astype('float32')

        min_max_aspect_ratios_order = draw(st.sampled_from([True, False]))
        clip = draw(st.sampled_from([True, False]))
        variances = [0.1, 0.1, 0.2, 0.2]  #common value

        fixed_sizes_1 = draw(
            st.lists(
                st.floats(
                    min_value=1, max_value=80), min_size=3, max_size=3))
        densities_1 = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=3, max_size=3))
        fixed_ratios_1 = draw(
            st.lists(
                st.floats(
                    min_value=1, max_value=10), min_size=1, max_size=1))

        fixed_sizes_2 = draw(
            st.lists(
                st.floats(
                    min_value=1, max_value=80), min_size=2, max_size=2))
        densities_2 = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=2, max_size=2))
        fixed_ratios_2 = draw(
            st.lists(
                st.floats(
                    min_value=1, max_value=10), min_size=1, max_size=1))

        value = draw(st.floats(min_value=1, max_value=1))
        step_h_1 = float(image_h) / float(density_prior_box_layer1_h)
        step_w_1 = float(image_w) / float(density_prior_box_layer1_w)
        step_h_2 = float(image_h) / float(density_prior_box_layer2_h)
        step_w_2 = float(image_w) / float(density_prior_box_layer2_w)
        offset_1 = 0.5  #common value
        offset_2 = 0.5  #common value

        #reshape2 params
        reshape2_w = draw(st.integers(min_value=4, max_value=4))
        #density_prior_box InferShape
        num_priors_1 = 0
        for index in range(len(densities_1)):
            num_priors_1 += len(fixed_ratios_1) * (pow(densities_1[index], 2))
        density_prior_box_out1_shape = [
            density_prior_box_layer1_h, density_prior_box_layer1_w,
            num_priors_1, 4
        ]  #4 is a common value
        num_priors_2 = 0
        for index in range(len(densities_2)):
            num_priors_2 += len(fixed_ratios_2) * (pow(densities_2[index], 2))
        density_prior_box_out2_shape = [
            density_prior_box_layer2_h, density_prior_box_layer2_w,
            num_priors_2, 4
        ]  #4 is a common value
        density_prior_box_layer1_length = density_prior_box_layer1_h * density_prior_box_layer1_w * num_priors_1
        density_prior_box_layer2_length = density_prior_box_layer2_h * density_prior_box_layer2_w * num_priors_2
        priorbox_shape = [
            density_prior_box_layer1_length + density_prior_box_layer2_length,
            4
        ]
        #box_coder params
        code_type = draw(st.sampled_from(["decode_center_size"]))
        axis = draw(st.sampled_from([0, 1]))
        box_normalized = draw(st.booleans())
        variance = draw(st.sampled_from([[0.1, 0.2, 0.3, 0.4], []]))
        lod_data = [[1, 1, 1, 1, 1]]

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

        density_prior_box_op_1 = OpConfig(
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
                "min_max_aspect_ratios_order": min_max_aspect_ratios_order,
                "variances": variances,
                "step_h_1": step_h_1,
                "step_w_1": step_w_1,
                "offset": offset_1,
                "fixed_sizes": fixed_sizes_1,
                "fixed_ratios": fixed_ratios_1,
                "densities": densities_1,
                "flatten_to_2d": False
            })

        density_prior_box_op_2 = OpConfig(
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
                "min_max_aspect_ratios_order": min_max_aspect_ratios_order,
                "variances": variances,
                "step_h_2": step_h_2,
                "step_w_2": step_w_2,
                "offset": offset_2,
                "fixed_sizes": fixed_sizes_2,
                "fixed_ratios": fixed_ratios_2,
                "densities": densities_2,
                "flatten_to_2d": False
            })

        reshape2_op_1 = OpConfig(
            type="reshape2",
            inputs={"X": ["density_prior_box_output_boxes_1"]},
            outputs={
                "Out": ["reshape2_out_1"],
                "XShape": ["reshape2_xshape_1"]
            },
            attrs={"shape": [-1, reshape2_w]})

        reshape2_op_2 = OpConfig(
            type="reshape2",
            inputs={"X": ["density_prior_box_output_boxes_2"]},
            outputs={
                "Out": ["reshape2_out_2"],
                "XShape": ["reshape2_xshape_2"]
            },
            attrs={"shape": [-1, reshape2_w]})

        reshape2_op_3 = OpConfig(
            type="reshape2",
            inputs={"X": ["density_prior_box_output_variances_1"]},
            outputs={
                "Out": ["reshape2_out_3"],
                "XShape": ["reshape2_xshape_3"]
            },
            attrs={"shape": [-1, reshape2_w]})

        reshape2_op_4 = OpConfig(
            type="reshape2",
            inputs={"X": ["density_prior_box_output_variances_2"]},
            outputs={
                "Out": ["reshape2_out_4"],
                "XShape": ["reshape2_xshape_4"]
            },
            attrs={"shape": [-1, reshape2_w]})

        axis_1 = draw(st.sampled_from([0]))
        axis_2 = draw(st.sampled_from([0]))

        concat_op_1 = OpConfig(
            type="concat",
            inputs={"X": ["reshape2_out_1", "reshape2_out_2"]},
            outputs={"Out": ["concat_out_1"]},
            attrs={"axis": axis_1})

        concat_op_2 = OpConfig(
            type="concat",
            inputs={"X": ["reshape2_out_3", "reshape2_out_4"]},
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
            density_prior_box_op_1, density_prior_box_op_2, reshape2_op_1,
            reshape2_op_2, reshape2_op_3, reshape2_op_4, concat_op_1,
            concat_op_2, box_coder_op
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
                    data_gen=partial(generate_targetbox), lod=lod_data),
            },
            outputs=["outputbox_data"])
        return program_config
    else:
        #image params
        batch_size = draw(st.integers(min_value=1, max_value=4))
        image_channels = 3
        image_w = draw(st.integers(min_value=40, max_value=50))
        image_h = draw(st.integers(min_value=40, max_value=50))

        density_prior_box_layer1_channels = draw(
            st.integers(
                min_value=30, max_value=40))
        density_prior_box_layer1_w = draw(
            st.integers(
                min_value=30, max_value=80))
        density_prior_box_layer1_h = draw(
            st.integers(
                min_value=30, max_value=80))

        density_prior_box_layer2_channels = draw(
            st.integers(
                min_value=30, max_value=40))
        density_prior_box_layer2_w = draw(
            st.integers(
                min_value=30, max_value=80))
        density_prior_box_layer2_h = draw(
            st.integers(
                min_value=30, max_value=80))

        def generate_image(*args, **kwargs):
            return np.random.random((batch_size, image_channels, image_w,
                                     image_h)).astype('float32')

        def generate_density_prior_box1(*args, **kwargs):
            return np.random.random(
                (batch_size, density_prior_box_layer1_channels,
                 density_prior_box_layer1_w,
                 density_prior_box_layer1_h)).astype('float32')

        def generate_density_prior_box2(*args, **kwargs):
            return np.random.random(
                (batch_size, density_prior_box_layer2_channels,
                 density_prior_box_layer2_w,
                 density_prior_box_layer2_h)).astype('float32')

        min_sizes = [2.0, 4.0]
        max_sizes = [5.0, 10.0]
        aspect_ratios = [2.0, 3.0]
        variances = [0.1, 0.1, 0.2, 0.2]

        flip = draw(st.sampled_from([True, False]))
        clip = draw(st.sampled_from([True, False]))
        min_max_aspect_ratios_order = draw(st.sampled_from([True, False]))

        value = draw(st.floats(min_value=1, max_value=1))
        step_h_1 = float(image_h) / float(density_prior_box_layer1_h)
        step_w_1 = float(image_w) / float(density_prior_box_layer1_w)
        step_h_2 = float(image_h) / float(density_prior_box_layer2_h)
        step_w_2 = float(image_w) / float(density_prior_box_layer2_w)
        offset_1 = 0.5  #common value
        offset_2 = 0.5  #common value

        #reshape2 params
        reshape2_w = draw(st.integers(min_value=4, max_value=4))
        #density_prior_box InferShape
        output_aspect_ratior = []
        prior_box_expand_aspect_ratios(
            aspect_ratios, flip, output_aspect_ratior)  #compute output size
        num_priors_1 = len(output_aspect_ratior) * len(
            min_sizes)  #3 need compute
        num_priors_1 += len(max_sizes)
        density_prior_box_out1_shape = [
            density_prior_box_layer1_h, density_prior_box_layer1_w,
            num_priors_1, 4
        ]  #4 is a common value
        num_priors_2 = len(output_aspect_ratior) * len(min_sizes)
        num_priors_2 += len(max_sizes)
        density_prior_box_out2_shape = [
            density_prior_box_layer2_h, density_prior_box_layer2_w,
            num_priors_2, 4
        ]  #4 is a common value
        density_prior_box_layer1_length = density_prior_box_layer1_h * density_prior_box_layer1_w * num_priors_1
        density_prior_box_layer2_length = density_prior_box_layer2_h * density_prior_box_layer2_w * num_priors_2
        priorbox_shape = [
            density_prior_box_layer1_length + density_prior_box_layer2_length,
            4
        ]
        #box_coder params
        code_type = draw(st.sampled_from(["decode_center_size"]))
        axis = draw(st.sampled_from([0, 1]))
        box_normalized = draw(st.booleans())
        variance = draw(st.sampled_from([[0.1, 0.2, 0.3, 0.4], []]))
        lod_data = [[1, 1, 1, 1, 1]]

        if code_type == "encode_center_size":
            targetbox_shape = draw(st.sampled_from([[30, 4], [80, 4]]))
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
            print(targetbox_shape)

        def generate_targetbox(*args, **kwargs):
            return np.random.random(targetbox_shape).astype(np.float32)

        density_prior_box_op_1 = OpConfig(
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

        density_prior_box_op_2 = OpConfig(
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

        reshape2_op_1 = OpConfig(
            type="reshape2",
            inputs={"X": ["density_prior_box_output_boxes_1"]},
            outputs={
                "Out": ["reshape2_out_1"],
                "XShape": ["reshape2_xshape_1"]
            },
            attrs={"shape": [-1, reshape2_w]})

        reshape2_op_2 = OpConfig(
            type="reshape2",
            inputs={"X": ["density_prior_box_output_boxes_2"]},
            outputs={
                "Out": ["reshape2_out_2"],
                "XShape": ["reshape2_xshape_2"]
            },
            attrs={"shape": [-1, reshape2_w]})

        reshape2_op_3 = OpConfig(
            type="reshape2",
            inputs={"X": ["density_prior_box_output_variances_1"]},
            outputs={
                "Out": ["reshape2_out_3"],
                "XShape": ["reshape2_xshape_3"]
            },
            attrs={"shape": [-1, reshape2_w]})

        reshape2_op_4 = OpConfig(
            type="reshape2",
            inputs={"X": ["density_prior_box_output_variances_2"]},
            outputs={
                "Out": ["reshape2_out_4"],
                "XShape": ["reshape2_xshape_4"]
            },
            attrs={"shape": [-1, reshape2_w]})

        axis_1 = draw(st.sampled_from([0]))
        axis_2 = draw(st.sampled_from([0]))

        concat_op_1 = OpConfig(
            type="concat",
            inputs={"X": ["reshape2_out_1", "reshape2_out_2"]},
            outputs={"Out": ["concat_out_1"]},
            attrs={"axis": axis_1})

        concat_op_2 = OpConfig(
            type="concat",
            inputs={"X": ["reshape2_out_3", "reshape2_out_4"]},
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
            density_prior_box_op_1, density_prior_box_op_2, reshape2_op_1,
            reshape2_op_2, reshape2_op_3, reshape2_op_4, concat_op_1,
            concat_op_2, box_coder_op
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
