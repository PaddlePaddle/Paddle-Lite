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

import numpy as np
import unittest
import abc
import os
import sys
import platform
import enum
import time
import logging
import shutil
import paddle
import paddle.fluid as fluid
import global_var_model as gl
from paddle.fluid.initializer import NumpyArrayInitializer
from paddle.fluid.core import PassVersionChecker
import paddle.fluid.core as core
from paddle import compat as cpt
import paddle.inference as paddle_infer
from typing import Optional, List, Callable, Dict, Any, Set
from program_config import TensorConfig, OpConfig, ProgramConfig, create_fake_model, create_quant_model

from itertools import product
from program_config import CxxConfig, TargetType, PrecisionType, DataLayoutType, Place

import hypothesis
from hypothesis import given, settings, seed, Verbosity
import hypothesis.strategies as st
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--target",
    choices=[
        'Host', 'X86', 'CUDA', 'ARM', 'OpenCL', 'FPGA', 'NPU', 'XPU', 'BM',
        'MLU', 'RKNPU', 'APU', 'HUAWEI_ASCEND_NPU', 'IMAGINATION_NNA',
        'INTEL_FPGA', 'Metal', 'NNAdapter'
    ],
    required=True)
parser.add_argument(
    "--enforce_rpc", default='off', type=str, help="whther rpc is enforced")

parser.add_argument(
    "--server_ip",
    default="localhost",
    type=str,
    help="when rpc is used , the ip address of the server")

parser.add_argument(
    "--url",
    type=str,
    help="Address of model download in model test", )
parser.add_argument(
    "--file_name",
    type=str,
    help="File name of the compressed model package downloaded in the model test",
)
parser.add_argument(
    "--model_name",
    type=str,
    help="Model name(That is, the prefix of the compressed package) in model test",
)
parser.add_argument(
    "--input_shapes", help="The tested model's input_shapes", action="append")
parser.add_argument(
    "--nnadapter_device_names",
    default="",
    type=str,
    help="Set nnadapter device names")
parser.add_argument(
    "--nnadapter_context_properties",
    default="",
    type=str,
    help="Set nnadapter context properties")
parser.add_argument(
    "--nnadapter_model_cache_dir",
    default="",
    type=str,
    help="Set nnadapter model cache dir")
parser.add_argument(
    "--nnadapter_subgraph_partition_config_path",
    default="",
    type=str,
    help="Set nnadapter subgraph partition config path")
parser.add_argument(
    "--nnadapter_mixed_precision_quantization_config_path",
    default="",
    type=str,
    help="Set nnadapter mixed precision quantization config path")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(message)s")

settings.register_profile(
    "ci",
    max_examples=200,
    suppress_health_check=hypothesis.HealthCheck.all(),
    deadline=None,
    print_blob=True,
    derandomize=True,
    report_multiple_bugs=False)

settings.register_profile(
    "ce",
    max_examples=1000,
    suppress_health_check=hypothesis.HealthCheck.all(),
    deadline=None,
    print_blob=True,
    derandomize=True,
    report_multiple_bugs=False)


class IgnoreReasonsBase(enum.Enum):
    # Paddle cannot predict normally (an error is reported in the prediction process) -- Both paddle and Lite do not predict
    PADDLE_NOT_SUPPORT = 0
    # Lite does not have a corresponding operator or Lite predicts an error -- Paddle predicts but Lite does not.
    PADDLELITE_NOT_SUPPORT = 1
    # When diff exists in the calculation results of Paddle and Lite -- Both Paddle and Lite predict,
    # but do not compare the results.
    ACCURACY_ERROR = 2
    #For ignore op fusion
    OP_FUSION_ERROR = 3


class AutoScanBaseTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.valid_places = []
        self.thread_num = [1]

        np.random.seed(1024)
        paddle.enable_static()
        super(AutoScanBaseTest, self).__init__(*args, **kwargs)
        self.ignore_cases = []
        abs_dir = os.path.abspath(os.path.dirname(__file__))
        self.cache_dir = os.path.join(abs_dir,
                                      str(self.__module__) + '_cache_dir')
        self.available_passes_in_framework = set()
        args = parser.parse_args()
        self.args = args
        self.vaild_nnadapter_device_names = []

    @abc.abstractmethod
    def sample_program_configs(self, draw):
        '''
        Generate all config with the combination of different Input tensor shape and
        different Attr values.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_input_data(self, draw):
        raise NotImplementedError

    @abc.abstractmethod
    def sample_predictor_configs(self):
        raise NotImplementedError

    @abc.abstractmethod
    def add_ignore_check_case(
            self,
            teller: [Callable[[ProgramConfig, CxxConfig], bool]],
            reason: IgnoreReasonsBase,
            note: str):
        self.ignore_cases.append((teller, reason, note))

    @abc.abstractmethod
    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def run_test_config(self, model, params, pred_config,
                        feed_data) -> Dict[str, np.ndarray]:
        '''
        Test a single case.
        '''
        pred_config.set_model_buffer(model, len(model), params, len(params))
        predictor = paddle_infer.create_predictor(pred_config)
        self.available_passes_in_framework = self.available_passes_in_framework | set(
            pred_config.pass_builder().all_passes())

        for name in feed_data:
            input_tensor = predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(feed_data[name]['data'])
            if feed_data[name]['lod'] is not None:
                input_tensor.set_lod(feed_data[name]['lod'])
        predictor.run()
        result = {}
        for o_name in predictor.get_output_names():
            result[o_name] = predictor.get_output_handle(o_name).copy_to_cpu()
        return result

    # count FP16 precision diff
    def count_fp16_diff(self, arr, base, atol, rtol) -> bool:
        diff = abs(arr - base)
        if len(diff) <= 0:
            pass
        max_diff = max(diff)
        check = False
        max_val = 0.0
        max_index = 0
        if max_diff > atol:
            print("max_diff: ", max_diff)
            size = len(arr)
            count = 0
            check = False
            for i in range(size):
                rel_val = diff[i] / max(arr[i], base[i])
                if (diff[i] > 1e-1 and abs(rel_val > rtol)):
                    if max_val < rel_val:
                        max_val = rel_val
                        max_index = i
                    check = True
            if check:
                print("max_val and max_index: ", max_val, max_index)
                print("value: ", base[max_index], arr[max_index],
                      diff[max_index])
                print("FP16 Output has diff. ")
                return False
            else:
                return True
        return True

    # count shape diff and data diff
    def count_shape_and_diff(self, base, arr, atol, rtol, flag_precision_fp16):
        base_shape = base.shape
        arr_shape = arr.shape
        base_len = len(base_shape)
        arr_len = len(arr_shape)
        diff_len = abs(base_len - arr_len)
        if diff_len == 1:
            # base=[1, K], arr=[k]
            if base_len > arr_len and (base_shape[0] == 1 or
                                       base_shape[-1] == 1):
                new_shape = base_shape[0:base_len - 1]
                if base_shape[0] == 1:
                    for i in range(1, base_len):
                        if i != 0:
                            new_shape[i - 1] = base_shape[i]

                self.assertTrue(
                    new_shape == arr_shape,
                    "The output shapes are not equal, the baseline shape is " +
                    str(new_shape) + ', but got ' + str(arr.shape))
                if flag_precision_fp16:
                    # count diff
                    arr_value = arr.flatten()
                    base_value = base.flatten()
                    # return true: has diff
                    res = self.count_fp16_diff(arr_value, base_value, atol,
                                               rtol)
                    self.assertTrue(res, "Output has diff. ")
                else:
                    diff = abs(base - arr)
                    self.assertTrue(
                        np.allclose(
                            base.flatten(),
                            arr.flatten(),
                            atol=atol,
                            rtol=rtol),
                        "Output has diff, max_diff : {}, index : {}.\nbase={}, \narr={}".
                        format(diff.max(), diff.argmax(), base, arr))
            # arr=[1, K], base=[k]
            elif base_len < arr_len and (arr_shape[0] == 1 or
                                         arr_shape[-1] == 1):
                new_shape = arr_shape[0:arr_len - 1]
                if arr_shape[0] == 1:
                    for i in range(1, arr_len):
                        new_shape[i - 1] = arr_shape[i]
                self.assertTrue(
                    new_shape == base_shape,
                    "The output shapes are not equal, the baseline shape is " +
                    str(base.shape) + ', but got ' + str(new_shape))

                if flag_precision_fp16:
                    # count diff
                    arr_value = arr.flatten()
                    base_value = base.flatten()
                    # return true: has diff
                    res = self.count_fp16_diff(arr_value, base_value, atol,
                                               rtol)
                    self.assertTrue(res, "Output has diff. ")
                else:
                    diff = abs(base - arr)
                    self.assertTrue(
                        np.allclose(
                            base.flatten(),
                            arr.flatten(),
                            atol=atol,
                            rtol=rtol),
                        "Output has diff, max_diff : {}, index : {}.\nbase={}, \narr={}".
                        format(diff.max(), diff.argmax(), base, arr))
            else:
                self.assertTrue(
                    base.shape == arr.shape,
                    "The output shapes are not equal, the baseline shape is " +
                    str(base.shape) + ', but got ' + str(arr.shape))
        else:
            self.assertTrue(
                base.shape == arr.shape,
                "The output shapes are not equal, the baseline shape is " +
                str(base.shape) + ', but got ' + str(arr.shape))
            if flag_precision_fp16:
                # count diff
                arr_value = arr.flatten()
                base_value = base.flatten()
                # return False: has diff
                res = self.count_fp16_diff(arr_value, base_value, atol, rtol)
                self.assertTrue(res, "Output has diff. ")
            else:
                diff = abs(base - arr)
                if diff.size != 0:
                    self.assertTrue(
                        np.allclose(
                            base, arr, atol=atol, rtol=rtol),
                        "Output has diff, max_diff : {}, index : {}.\nbase={}, \narr={}".
                        format(diff.max(), diff.argmax(), base, arr))
                else:
                    self.assertTrue(
                        np.allclose(
                            base, arr, atol=atol, rtol=rtol),
                        "Output has diff,\nbase={}, \narr={}".format(base,
                                                                     arr))

    @abc.abstractmethod
    def assert_tensors_near(self,
                            atol: float,
                            rtol: float,
                            tensor: Dict[str, np.array],
                            baseline: Dict[str, np.array],
                            flag_precision_fp16: False):
        if len(tensor) == 0 and len(baseline) == 0:
            return
        if len(tensor) == 1 and len(baseline) == 1:
            tensor_key = list(tensor.keys())
            arr = np.array(tensor[tensor_key[0]])
            base_key = list(baseline.keys())
            base = np.array(baseline[base_key[0]])

            if not base.shape and arr.shape == (1, ):
                pass
            else:
                self.count_shape_and_diff(base, arr, atol, rtol,
                                          flag_precision_fp16)
        else:
            for key in tensor:
                suffix_str = [
                    "/target_trans", "__Mangled_1", "/precision_trans"
                ]
                paddlekey = key
                for s_str in suffix_str:
                    index = key.rfind(s_str)
                    if index > 0:
                        paddlekey = key[0:index]

                if (paddlekey == "saved_mean" or
                        paddlekey == "saved_variance" or
                        paddlekey == "mean_data" or
                        paddlekey == "variance_data"):
                    # training using data
                    continue
                arr = np.array(tensor[key])
                self.count_shape_and_diff(baseline[paddlekey], arr, atol, rtol,
                                          flag_precision_fp16)

    def generate_op_config(self,
                           ops_config: List[Dict[str, Any]]) -> List[OpConfig]:
        ops = []
        for i in range(len(ops_config)):
            op_config = ops_config[i]
            ops.append(
                OpConfig(
                    type=op_config['op_type'],
                    inputs=op_config['op_inputs'],
                    outputs=op_config['op_outputs'],
                    attrs=op_config['op_attrs']))
        return ops

    @abc.abstractmethod
    def ignore_log(self, msg: str):
        logging.warning("SKIP: " + msg)

    @abc.abstractmethod
    def fail_log(self, msg: str):
        logging.fatal("FAILE: " + msg)

    @abc.abstractmethod
    def success_log(self, msg: str):
        logging.info("SUCCESS: " + msg)

    @abc.abstractmethod
    def is_model_test(self) -> bool:
        return False

    @abc.abstractmethod
    def get_model(self, draw):
        raise NotImplementedError

    @abc.abstractmethod
    def create_inference_config(self,
                                passes: Optional[List[str]]=None,
                                use_gpu: bool=False,
                                use_mkldnn: bool=False,
                                ir_optim: Optional[bool]=None):
        config = paddle_infer.Config()
        config.switch_ir_debug(True)
        config.disable_glog_info()
        if ir_optim is not None:
            config.switch_ir_optim(ir_optim)
        if use_gpu:
            config.enable_use_gpu(100, 0)
        if use_mkldnn:
            config.enable_mkldnn()
        if passes is not None:
            config.pass_builder().set_passes(passes)
        return config

    @abc.abstractmethod
    def insert_leaky_relu_op(self, prog_configs=None):
        alpha_data = 0.01
        leaky_relu_op = OpConfig(
            type="leaky_relu",
            inputs={"X": []},
            outputs={"Out": ["output_act_data"]},
            attrs={"alpha": alpha_data})
        leaky_relu_op.inputs["X"].append(prog_configs.outputs[0])
        prog_configs.ops.append(leaky_relu_op)
        prog_configs.outputs[0] = "output_act_data"

    def run_test(self, quant=False, prog_configs=None):
        status = True

        paddlelite_configs, op_list_, (atol_,
                                       rtol_) = self.sample_predictor_configs()
        for prog_config in prog_configs:

            predictor_idx = -1
            cnt = 0
            for paddlelite_config in paddlelite_configs:
                flag_precision_fp16 = False
                if paddlelite_config.precision(
                ) == PrecisionType.FP16 and paddlelite_config.target(
                ) == TargetType.ARM:
                    flag_precision_fp16 = True
                if paddlelite_config.precision(
                ) == PrecisionType.INT8 and paddlelite_config.target(
                ) == TargetType.ARM:
                    quant = True
                else:
                    quant = False

                predictor_idx += 1
                # judge validity of program
                if not self.is_program_valid(prog_config, paddlelite_config):
                    self.num_invalid_programs_list[predictor_idx] += 1
                    continue
                self.num_ran_programs_list[predictor_idx] += 1

                if flag_precision_fp16:
                    if platform.system() == 'Linux':
                        # only run in M1
                        continue
                # creat model and prepare feed data
                if flag_precision_fp16:
                    atol_ = 1e-1
                    rtol_ = 5.3e-2
                if quant:
                    if platform.system() == 'Darwin' or platform.processor(
                    ) == 'x86_64':
                        # only run in linux
                        continue
                    atol_ = 8e-3
                    rtol_ = 8e-3
                    if cnt == 0:
                        self.insert_leaky_relu_op(prog_config)
                    cnt = cnt + 1
                    model, params = create_fake_model(prog_config)
                    model, params = create_quant_model(
                        model, params, self.cache_dir, prog_config)
                else:
                    model, params = create_fake_model(prog_config)

                feed_data = {}
                for name, tensor_config in prog_config.inputs.items():
                    feed_data[name] = {
                        'data': tensor_config.data,
                        'lod': tensor_config.lod
                    }
                results: List[Dict[str, np.ndarray]] = []
                # ignore info
                accuracy_error_flag = False
                paddle_not_support_flag = False
                paddle_lite_not_support_flag = False
                op_fusion_error_flag = False
                pred_config = paddlelite_config.value()
                for ignore_info in self.ignore_cases:
                    if ignore_info[0](prog_config, paddlelite_config):
                        self.num_ignore_tests_list[predictor_idx] += 1
                        if ignore_info[1] == IgnoreReasonsBase.ACCURACY_ERROR:
                            accuracy_error_flag = True
                            self.ignore_log("[ACCURACY_ERROR] " + ignore_info[
                                2] + ' ' + ' vs ' + self.paddlelite_config_str(
                                    pred_config))
                            gl.set_out_diff_ops(
                                self.get_target(),
                                self.get_nnadapter_device_name(), sys.argv[0])
                        elif ignore_info[
                                1] == IgnoreReasonsBase.PADDLELITE_NOT_SUPPORT:
                            paddle_lite_not_support_flag = True
                            self.ignore_log("[PADDLELITE_NOT_SUPPORT ERROR] " +
                                            ignore_info[2] + ' ' + ' vs ' +
                                            self.paddlelite_config_str(
                                                pred_config))
                        elif ignore_info[
                                1] == IgnoreReasonsBase.PADDLE_NOT_SUPPORT:
                            paddle_not_support_flag = True
                            self.ignore_log("[PADDLE_NOT_SUPPORT ERROR] " +
                                            ignore_info[2] + ' ' + ' vs ' +
                                            self.paddlelite_config_str(
                                                pred_config))
                        elif ignore_info[
                                1] == IgnoreReasonsBase.OP_FUSION_ERROR:
                            op_fusion_error_flag = True
                            self.ignore_log("[OP_FUSION ERROR] " + ignore_info[
                                2] + ' ' + ' vs ' + self.paddlelite_config_str(
                                    pred_config))
                        else:
                            raise NotImplementedError

                if paddle_not_support_flag:
                    gl.set_paddle_not_supported_ops(
                        self.get_target(),
                        self.get_nnadapter_device_name(), sys.argv[0])
                    continue

                # baseline: cpu no ir_optim run
                base_config = self.create_inference_config(ir_optim=False)
                logging.info('[ProgramConfig]: ' + str(prog_config))
                results.append(
                    self.run_test_config(model, params, base_config,
                                         feed_data))
                if paddle_lite_not_support_flag:
                    gl.set_lite_not_supported_ops(
                        self.get_target(),
                        self.get_nnadapter_device_name(), sys.argv[0])
                    continue

                if os.path.exists(self.cache_dir):
                    shutil.rmtree(self.cache_dir)
                if not os.path.exists(self.cache_dir):
                    os.mkdir(self.cache_dir)
                try:
                    result, opt_model_bytes = self.run_lite_config(
                        model, params, feed_data, pred_config, args.server_ip)
                    results.append(result)
                    # add ignore methods
                    if self.passes is not None:  # pass check
                        if not accuracy_error_flag:
                            self.assert_tensors_near(atol_, rtol_, results[-1],
                                                     results[0],
                                                     flag_precision_fp16)
                        if not op_fusion_error_flag:
                            self.assert_op_list(opt_model_bytes, op_list_)
                    else:  # op check
                        self.assert_kernel_type(opt_model_bytes, op_list_,
                                                paddlelite_config)
                        if not accuracy_error_flag:
                            self.assert_tensors_near(atol_, rtol_, results[-1],
                                                     results[0],
                                                     flag_precision_fp16)
                except Exception as e:
                    self.fail_log(
                        self.paddlelite_config_str(pred_config) +
                        '\033[1;31m \nERROR INFO: {}\033[0m'.format(str(e)))
                    status = False
                    break
                self.success_log('PredictorConfig: ' +
                                 self.paddlelite_config_str(pred_config))
        self.assertTrue(status)
        gl.set_success_ops(self.get_target(),
                           self.get_nnadapter_device_name(), sys.argv[0])

    def run_model_test(self, inputs_configs=None, model=None, params=None):
        status = True
        paddlelite_configs, _, (atol_, rtol_) = self.sample_predictor_configs()
        for inputs_config in inputs_configs:
            feed_data = {}
            for name, tensor_config in inputs_config.items():
                feed_data[name] = {
                    'data': tensor_config.data,
                    'lod': tensor_config.lod
                }

            results: List[Dict[str, np.ndarray]] = []

            # baseline: cpu no ir_optim run
            base_config = self.create_inference_config(ir_optim=False)
            results.append(
                self.run_test_config(model, params, base_config, feed_data))
            flag_precision_fp16 = False
            for paddlelite_config in paddlelite_configs:
                pred_config = paddlelite_config.value()

                try:
                    result, opt_model_bytes = self.run_lite_config(
                        model, params, feed_data, pred_config, args.server_ip)
                    results.append(result)
                    self.assert_tensors_near(atol_, rtol_, results[-1],
                                             results[0], flag_precision_fp16)
                except Exception as e:
                    self.fail_log(
                        self.paddlelite_config_str(pred_config) +
                        '\033[1;31m \nERROR INFO: {}\033[0m'.format(str(e)))
                    status = False
                    break
                self.success_log('PredictorConfig: ' +
                                 self.paddlelite_config_str(pred_config))
            self.assertTrue(status)

    def inference_config_str(self, config) -> bool:
        dic = {}
        enable_mkldnn = config.mkldnn_enabled()
        dic['use_mkldnn'] = enable_mkldnn
        enable_gpu = config.use_gpu()
        return str(dic)

    def paddlelite_config_str(self, config) -> bool:
        return str(config)

    # method for ignoring
    def add_ignore_pass_case(self):
        return

    # judge if program contain op_list
    def assert_op_list(self, model_bytes, op_list_after_fusion):
        if not self.passes:
            raise ValueError(
                "In PassAutoScan you should give a valid pass name.")
        pg = paddle.static.deserialize_program(model_bytes)
        main_block = pg.desc.block(0)
        after_op_list = list()
        for i in range(main_block.op_size()):
            if main_block.op(i).type(
            ) in ["feed", "fetch", "layout", "io_copy"]:
                continue
            after_op_list.append(main_block.op(i).type())
        self.assertTrue(
            op_list_after_fusion == after_op_list,
            "Expected operator list after fusion is {}, but now it's {}".
            format(op_list_after_fusion, after_op_list), )

    # judge if correct kernel is picked
    def assert_kernel_type(self, model_bytes, op_list, paddlelite_config):
        pg = paddle.static.deserialize_program(model_bytes)
        main_block = pg.desc.block(0)
        after_op_list = list()
        target_ = paddlelite_config.target()
        precision_ = paddlelite_config.precision()
        layout_ = paddlelite_config.layout()

        for i in range(main_block.op_size()):
            if main_block.op(i).type() in op_list:
                kernel_type_info = main_block.op(i).attr(
                    "__@kernel_type_attr@__").split("/")
                self.assertTrue(
                    len(kernel_type_info) == 5,
                    "Incompleted kernel info of {}:{}".format(
                        main_block.op(i).type(),
                        main_block.op(i).attr("__@kernel_type_attr@__")))
                current_target_ = TargetType(int(kernel_type_info[2]))
                current_precision_ = PrecisionType(int(kernel_type_info[3]))
                current_layout_ = DataLayoutType(int(kernel_type_info[4]))
                correct_kernel_flag_ = (target_ == current_target_) and (
                    precision_ == current_precision_ or
                    current_precision_ == PrecisionType.Any) and (
                        layout_ == current_layout_ or
                        current_layout_ == DataLayoutType.Any)
                self.assertTrue(
                    correct_kernel_flag_ == True,
                    "Expected kernel_type of op {} is ({},{},{}), but now it's ({},{},{})".
                    format(
                        main_block.op(i).type(), target_, precision_, layout_,
                        current_target_, current_precision_, current_layout_))

    def run_and_statis(
            self,
            quant=False,
            max_examples=100,
            reproduce=None,
            min_success_num=25,
            passes=None,
            model=None,
            params=None, ):
        self.init_statistical_parameters()
        settings.register_profile(
            "dev",
            max_examples=max_examples,
            suppress_health_check=hypothesis.HealthCheck.all(),
            deadline=None,
            print_blob=True,
            derandomize=True,
            report_multiple_bugs=False,
            verbosity=Verbosity.verbose)

        if os.getenv('HYPOTHESIS_TEST_PROFILE') == "ci":
            settings.load_profile("ci")
        elif os.getenv('HYPOTHESIS_TEST_PROFILE') == "ce":
            settings.load_profile("ce")
        else:
            settings.load_profile("dev")

        self.passes = passes
        self.add_ignore_pass_case()

        def program_generator(draw):
            return self.sample_program_configs(draw)

        def inputs_generator(draw):
            return self.prepare_input_data(draw)

        def run_test(prog_config):
            return self.run_test(quant=quant, prog_configs=[prog_config])

        def run_model_test(inputs_configs):
            return self.run_model_test(
                inputs_configs=[inputs_configs], model=model, params=params)

        # if current unittest is not active on the input targ    paddlelite_not_support_flag = Trueet, we will exit directly.
        gl.set_all_test_ops(self.get_target(),
                            self.get_nnadapter_device_name(), sys.argv[0])
        if not self.is_actived():
            logging.info("Error: This test is not actived on " +
                         self.get_target())
            return

        if not self.is_nnadapter_device_actived():
            logging.info("Error: This test is not actived on " +
                         self.get_nnadapter_device_name())
            return

        if self.get_target().upper() == "NNADAPTER":
            self.assertTrue(
                self.args.nnadapter_device_names != "",
                "Args Error: Please set nnadapter_device_names when target=nnadapter!"
            )

        if self.is_model_test():
            generator = st.composite(inputs_generator)
            loop_func = given(generator())(run_model_test)
        else:
            generator = st.composite(program_generator)
            loop_func = given(generator())(run_test)

        if reproduce is not None:
            loop_func = reproduce(loop_func)
        logging.info("Start to running test of {}".format(type(self)))
        loop_func()
        if self.is_model_test():
            logging.info(
                "===================Statistical Information===================")
            logging.info("Number of Input Configs: {}".format(max_examples))
            logging.info("Number of Predictor Kinds: {}".format(
                int(self.num_predictor_kinds)))
        else:
            logging.info(
                "===================Statistical Information===================")
            logging.info("Number of Generated Programs: {}".format(
                max_examples))
            logging.info("Number of Predictor Kinds: {}".format(
                int(self.num_predictor_kinds)))
            self.assertTrue(self.num_predictor_kinds > 0,
                            "Number of Predictor Kinds must be greater than 0")
            logging.info("Number of Ran Programs: {}".format(
                self.num_ran_programs_list))
            logging.info("Number of Invalid Programs: {}".format(
                self.num_invalid_programs_list))
            logging.info("Number of Ignored Tests: {}".format(
                self.num_ignore_tests_list))
            successful_ran_programs = int(
                (sum(self.num_ran_programs_list) +
                 sum(self.num_ignore_tests_list)) / self.num_predictor_kinds)

            logging.info(
                "Number of successfully ran programs approximately equal to {}".
                format(successful_ran_programs))
            if successful_ran_programs < min_success_num:
                logging.fatal(
                    "At least {} programs need to ran successfully, but now only about {} programs satisfied.".
                    format(min_success_num, successful_ran_programs))
                assert False

    @abc.abstractmethod
    def run_lite_config(self,
                        model,
                        params,
                        feed_data,
                        pred_config,
                        server_ip="localhost") -> Dict[str, np.ndarray]:
        raise NotImplementedError

    # enable a predictor config
    # configs will be generated automatically according to inputs
    def enable_testing_on_place(self,
                                target=None,
                                precision=None,
                                layout=None,
                                thread=None,
                                places=None) -> None:
        # set thread_num
        if isinstance(thread, list):
            self.thread_num = list(set(self.thread_num + thread))
        if isinstance(thread, int):
            self.thread_num.append(thread)
            self.thread_num = list(self.thread_num)

        # arm basic places:
        arm_basic_places = [
            Place(TargetType.ARM, PrecisionType.INT32),
            Place(TargetType.ARM, PrecisionType.INT64)
        ]

        # if list[Place] is inputed, this will be used directly
        if places is not None:
            assert isinstance(places, list)
            self.valid_places.append(places)
            return
        # otherwise we will generate a list[Place] from the inputed[target\precision\layout]
        assert (target is not None)
        target_ = target if isinstance(target, list) else [target]
        self.target = target_
        precision_ = precision if isinstance(precision, list) else [precision]
        layout_ = layout if isinstance(layout, list) else [layout]

        for tar_, pre_, lay_ in product(target_, precision_, layout_):
            if (tar_ == TargetType.ARM):
                self.valid_places.append([Place(tar_, pre_, lay_)] +
                                         arm_basic_places)
            else:
                self.valid_places.append([Place(tar_, pre_, lay_)])

    def enable_devices_on_nnadapter(self, device_names=list) -> None:
        self.vaild_nnadapter_device_names = device_names

    def get_target(self) -> str:
        return self.args.target

    def is_actived(self) -> bool:
        for valid_place_ in self.valid_places:
            if self.get_target() in valid_place_[0]:
                return True
        return False

    def is_nnadapter_device_actived(self) -> bool:
        if self.get_target().upper() != "NNADAPTER":
            return True
        if self.get_nnadapter_device_name(
        ) in self.vaild_nnadapter_device_names:
            return True
        return False

    def get_nnadapter_device_name(self) -> str:
        nnadapter_device_name_list = self.args.nnadapter_device_names.split(
            ",")
        return nnadapter_device_name_list[0]

    def get_predictor_configs(self) -> List[CxxConfig]:
        return self.target_to_predictor_configs(self, self.get_target())

    def init_statistical_parameters(self):
        self.num_predictor_kinds = len(self.get_predictor_configs())
        self.num_invalid_programs_list = [0] * self.num_predictor_kinds
        self.num_ran_programs_list = [0] * self.num_predictor_kinds
        self.num_ignore_tests_list = [0] * self.num_predictor_kinds

    @staticmethod
    def nnadapter_config_set(self, config: CxxConfig):
        config.set_nnadapter_device_names(
            self.args.nnadapter_device_names.split(","))
        config.set_nnadapter_context_properties(
            self.args.nnadapter_context_properties)
        config.set_nnadapter_model_cache_dir(
            self.args.nnadapter_model_cache_dir)
        config.set_nnadapter_subgraph_partition_config_path(
            self.args.nnadapter_subgraph_partition_config_path)
        config.set_nnadapter_mixed_precision_quantization_config_path(
            self.args.nnadapter_mixed_precision_quantization_config_path)

    # get valid test configs
    @staticmethod
    def target_to_predictor_configs(self, target: str) -> List[CxxConfig]:
        configs_ = []
        for elem_ in self.valid_places:
            if target in elem_[0]:
                for thread_ in self.thread_num:
                    config_ = CxxConfig()
                    config_.set_valid_places(elem_)
                    config_.set_threads(thread_)
                    if target.upper() == "NNADAPTER":
                        self.nnadapter_config_set(self, config_)
                    configs_.append(config_)
        return configs_
