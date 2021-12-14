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
import enum
import time
import logging
import shutil
import paddle
import paddle.fluid as fluid
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
from hypothesis import given, settings, seed
import hypothesis.strategies as st
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target", choices=['Host', 'X86','CUDA','ARM','OpenCL','FPGA','NPU','MLU','RKNPU','APU','HUAWEI_ASCEND_NPU','INTEL_FPGA'], required=True)
logging.basicConfig(level=logging.INFO, format="%(message)s")

settings.register_profile(
    "ci",
    max_examples=10,
    suppress_health_check=hypothesis.HealthCheck.all(),
    deadline=None,
    print_blob=True,
    derandomize=True,
    report_multiple_bugs=False)
settings.load_profile("ci")

class IgnoreReasonsBase(enum.Enum):
    # Paddle not support, but paddlelite support, we need to add the feature.
    PADDLE_NOT_IMPLEMENTED = 0
    # paddlelite not support.
    PADDLELITE_NOT_SUPPORT = 1
    # Accuracy is abnormal after enabling pass.
    ACCURACY_ERROR = 2



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
        self.num_ran_programs = 0
        self.num_invalid_programs = 0
        self.num_ignore_tests = 0
        self.num_predictor_kinds = 0

        args = parser.parse_args()
        self.args = args


    @abc.abstractmethod
    def sample_program_configs(self, draw):
        '''
        Generate all config with the combination of different Input tensor shape and
        different Attr values.
        '''
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
    def is_program_valid(self, program_config: ProgramConfig, predictor_config: CxxConfig) -> bool:
        return True

    def run_test_config(self, model, params, prog_config, pred_config,
                        feed_data) -> Dict[str, np.ndarray]:
        '''
        Test a single case.
        '''
        pred_config.set_model_buffer(model, len(model), params, len(params))
        predictor = paddle_infer.create_predictor(pred_config)
        self.available_passes_in_framework = self.available_passes_in_framework | set(
            pred_config.pass_builder().all_passes())

        for name, _ in prog_config.inputs.items():
            input_tensor = predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(feed_data[name]['data'])
            if feed_data[name]['lod'] is not None:
                input_tensor.set_lod(feed_data[name]['lod'])
        predictor.run()
        result = {}
        for out_name, o_name in zip(prog_config.outputs,
                                    predictor.get_output_names()):
            result[out_name] = predictor.get_output_handle(o_name).copy_to_cpu()
        return result


    @abc.abstractmethod
    def assert_tensors_near(self,
                            atol: float,
                            rtol: float,
                            tensor: Dict[str, np.array],
                            baseline: Dict[str, np.array]):
        if len(tensor) == 1 and len(baseline) == 1:
            tensor_key = list(tensor.keys())
            arr = np.array(tensor[tensor_key[0]])
            base_key = list(baseline.keys())
            base = np.array(baseline[base_key[0]])
            self.assertTrue(
                    base.shape == arr.shape,
                    "The output shapes are not equal, the baseline shape is " +
                    str(base.shape) + ', but got ' + str(arr.shape))
            self.assertTrue(
                np.allclose(
                    base, arr, atol=atol, rtol=rtol),
                "Output has diff. ")
        else:
            for key in tensor:
                opencl_str = "/target_trans"
                index = key.rfind(opencl_str)
                paddlekey=key
                if index > 0:
                    paddlekey = key[0: index]
                if (key == "saved_mean" or key == "saved_variance"):
                    # training using data
                    continue
                arr = np.array(tensor[key])
                self.assertTrue(
                    baseline[paddlekey].shape == arr.shape,
                    "The output shapes are not equal, the baseline shape is " +
                    str(baseline[paddlekey].shape) + ', but got ' + str(arr.shape))
                self.assertTrue(
                    np.allclose(
                        baseline[paddlekey], arr, atol=atol, rtol=rtol),
                    "Output has diff. ")

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
            self.passes = passes
        return config

    def run_test(self, quant=False, prog_configs=None):
        status = True

        paddlelite_configs, op_list_, (atol_, rtol_) = self.sample_predictor_configs()
        for prog_config in prog_configs:
            # if program is invalid, we should ignore this cases.
            program_valid_ = False
            for paddlelite_config in paddlelite_configs:
                # judge validity of program
                if self.is_program_valid(prog_config, paddlelite_config):
                    program_valid_ = True
            if not program_valid_:
                self.num_invalid_programs += 1
                continue


            self.num_ran_programs += 1
            model, params = create_fake_model(prog_config)
            if quant:
                model, params = create_quant_model(model, params)

            feed_data = {}
            for name, tensor_config in prog_config.inputs.items():
                feed_data[name] = {
                    'data': tensor_config.data,
                    'lod': tensor_config.lod
                }
            results: List[Dict[str, np.ndarray]] = []

            # baseline: cpu no ir_optim run
            base_config = self.create_inference_config(ir_optim=False)
            logging.info('[ProgramConfig]: ' + str(prog_config))
            results.append(
                self.run_test_config(model, params, prog_config, base_config,
                                     feed_data))


            for paddlelite_config in paddlelite_configs:
                # judge validity of program
                if not self.is_program_valid(prog_config, paddlelite_config):
                    continue

                self.num_predictor_kinds += 1
                # ignore info
                ignore_flag = False
                paddle_lite_not_support_flag = False
                pred_config = paddlelite_config.value()
                for ignore_info in self.ignore_cases:
                    if ignore_info[0](prog_config, paddlelite_config):
                        ignore_flag = True
                        self.num_ignore_tests += 1
                        if ignore_info[1] == IgnoreReasonsBase.ACCURACY_ERROR:
                            self.ignore_log("[ACCURACY_ERROR] " +
                                          ignore_info[2] + ' ' + ' vs ' + self.
                                          paddlelite_config_str(pred_config))
                        elif  ignore_info[1] == IgnoreReasonsBase.PADDLELITE_NOT_SUPPORT:
                            paddle_lite_not_support_flag = True
                            self.ignore_log("[PADDLELITE_NOT_SUPPORT ERROR] " +
                                          ignore_info[2] + ' ' + ' vs ' + self.
                                          paddlelite_config_str(pred_config))
                            break
                        else:
                            raise NotImplementedError
                        break
                if paddle_lite_not_support_flag:
                    continue

                if os.path.exists(self.cache_dir):
                    shutil.rmtree(self.cache_dir)
                if not os.path.exists(self.cache_dir):
                    os.mkdir(self.cache_dir)
                try:
                    result, opt_model_bytes = self.run_lite_config(model, params, feed_data, pred_config)
                    results.append(result)
                    if self.passes is not None:
                        # op unit test: we will not check precision in ignore case
                        self.assert_tensors_near(atol_, rtol_, results[-1],
                                                 results[0])
                        if not ignore_flag:
                            # pass unit test: we will not check fusion in ignore case
                            self.assert_op_list(opt_model_bytes, op_list_)
                except Exception as e:
                    self.fail_log(
                        self.paddlelite_config_str(pred_config) +
                        '\033[1;31m \nERROR INFO: {}\033[0m'.format(str(e)))
                    if not ignore_flag:
                        status = False
                    continue
                self.success_log('PredictorConfig: ' + self.
                                 paddlelite_config_str(pred_config))
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
            if main_block.op(i).type() in ["feed", "fetch"]:
                continue
            after_op_list.append(main_block.op(i).type())
        self.assertTrue(
            op_list_after_fusion == after_op_list,
            "Expected operator list after fusion is {}, but now it's {}".format(
                op_list_after_fusion, after_op_list), )


    def run_and_statis(
            self,
            quant=False,
            max_examples=100,
            reproduce=None,
            min_success_num=25,
            max_duration=180,
            passes=None ):
        if os.getenv('HYPOTHESIS_TEST_PROFILE', 'ci') == "dev":
            max_examples *= 10
            min_success_num *= 10
            # while at ce phase, there's no limit on time
            max_duration = -1
        start_time = time.time()
        settings.register_profile(
            "ci",
            max_examples=max_examples,
            suppress_health_check=hypothesis.HealthCheck.all(),
            deadline=None,
            print_blob=True,
            derandomize=True,
            report_multiple_bugs=False, )
        settings.load_profile("ci")

        self.passes = passes
        self.add_ignore_pass_case()

        def program_generator(draw):
            return self.sample_program_configs(draw)

        def run_test(prog_config):
            return self.run_test(quant=quant, prog_configs=[prog_config])

        # if current unittest is not active on the input targ    paddlelite_not_support_flag = Trueet, we will exit directly.
        if not self.is_actived():
            logging.info("Error: This test is not actived on " + self.get_target())
            return

        generator = st.composite(program_generator)
        loop_func = given(generator())(run_test)
        if reproduce is not None:
            loop_func = reproduce(loop_func)
        logging.info("Start to running test of {}".format(type(self)))
        loop_func()
        logging.info(
            "===================Statistical Information===================")
        logging.info("Number of Generated Programs: {}".format(
            self.num_ran_programs + self.num_invalid_programs))
        logging.info("Number of Invalid Programs: {}".format(
            self.num_invalid_programs))
        logging.info("Number of Ran Programs: {}".format(self.num_ran_programs))
        logging.info("Number of Ignored Tests: {}".format(
            self.num_ignore_tests))
        if self.num_predictor_kinds == 0:
            successful_ran_programs = int(self.num_ran_programs)
            min_success_num = 0
        else:
            successful_ran_programs = int(self.num_ran_programs -
                                        self.num_ignore_tests /
                                        self.num_predictor_kinds)

        logging.info(
            "Number of successfully ran programs approximately equal to {}".
            format(successful_ran_programs))
        if successful_ran_programs < min_success_num:
            logging.warning(
                "satisfied_programs = ran_programs - num_ignore_tests / num_predictor_kinds"
            )
            logging.fatal(
                "At least {} programs need to ran successfully, but now only about {} programs satisfied.".
                format(min_success_num, successful_ran_programs))
            assert False
        used_time = time.time() - start_time
        if max_duration > 0 and used_time > max_duration:
            logging.fatal(
                "The duration exceeds {} seconds, if this is neccessary, try to set a larger number for parameter `max_duration`.".
                format(max_duration))
            assert False

    @abc.abstractmethod
    def run_lite_config(self, model, params, feed_data, pred_config) -> Dict[str, np.ndarray]:
        raise NotImplementedError


    # enable a predictor config
    # configs will be generated automatically according to inputs
    def enable_testing_on_place(self, target=None, precision=None, layout=None, thread=None, places=None) -> None:
        # set thread_num
        if isinstance(thread,list):
            self.thread_num = list(set(self.thread_num + thread))
        if isinstance(thread,int):
            self.thread_num.append(thread)
            self.thread_num = list(self.thread_num)

        # if list[Place] is inputed, this will be used directly
        if places is not None:
            assert isinstance(places, list)
            self.valid_places.append(places)
            return
        # otherwise we will generate a list[Place] from the inputed[target\precision\layout]
        assert  (target is not None)
        target_ = target if isinstance(target,list) else [target]
        precision_ = precision if isinstance(precision, list) else [precision]
        layout_ = layout if isinstance(layout,list) else [layout]
        for tar_, pre_, lay_ in product(target_, precision_, layout_):
            self.valid_places.append([Place(tar_, pre_, lay_)])
        return


    def get_target(self) -> str:
        return self.args.target


    def is_actived(self) -> bool:
        for valid_place_ in self.valid_places:
            if self.get_target() in valid_place_[0]:
                return True
        return False

    def get_predictor_configs(self) -> List[CxxConfig]:
        return self.target_to_predictor_configs(self, self.get_target())

    # get valid test configs
    @staticmethod
    def target_to_predictor_configs(self,target:str) -> List[CxxConfig]:
       configs_ = []
       for elem_ in self.valid_places:
           if target in elem_[0]:
               for thread_ in self.thread_num:
                 config_ = CxxConfig()
                 config_.set_valid_places(elem_)
                 config_.set_threads(thread_)
                 configs_.append(config_)
       return configs_
