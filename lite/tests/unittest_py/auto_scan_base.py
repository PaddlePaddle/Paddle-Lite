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

import hypothesis
from hypothesis import given, settings, seed, example, assume

logging.basicConfig(level=logging.INFO, format="%(message)s")

settings.register_profile(
    "ci",
    max_examples=100,
    suppress_health_check=hypothesis.HealthCheck.all(),
    deadline=None,
    print_blob=True,
    derandomize=True,
    report_multiple_bugs=False)
settings.load_profile("ci")

class SkipReasonsBase(enum.Enum):
    # Paddle not support, but paddlelite support, we need to add the feature.
    PADDLE_NOT_IMPLEMENTED = 0
    # paddlelite not support.
    PADDLELITE_NOT_SUPPORT = 1
    # Accuracy is abnormal after enabling pass.
    ACCURACY_ERROR = 2



class AutoScanBaseTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        np.random.seed(1024)
        paddle.enable_static()
        super(AutoScanBaseTest, self).__init__(*args, **kwargs)
        self.skip_cases = []
        abs_dir = os.path.abspath(os.path.dirname(__file__))

    @abc.abstractmethod
    def sample_program_configs(self):
        '''
        Generate all config with the combination of different Input tensor shape and
        different Attr values.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def sample_predictor_configs(self):
        raise NotImplementedError

    @abc.abstractmethod
    def add_skip_case(
            self,
            teller: [Callable[[ProgramConfig, paddle_infer.Config], bool]],
            reason: SkipReasonsBase,
            note: str):
        self.skip_cases.append((teller, reason, note))

    @abc.abstractmethod
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        raise NotImplementedError

    def run_test_config(self, model, params, prog_config, pred_config,
                        feed_data) -> Dict[str, np.ndarray]:
        '''
        Test a single case.
        '''
        pred_config.set_model_buffer(model, len(model), params, len(params))
        predictor = paddle_infer.create_predictor(pred_config)

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
        for key, arr in tensor.items():
            self.assertTrue(
                baseline[key].shape == arr.shape,
                "The output shapes are not equal, the baseline shape is " +
                str(baseline[key].shape) + ', but got ' + str(arr.shape))
            self.assertTrue(
                np.allclose(
                    baseline[key], arr, atol=atol, rtol=rtol),
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
    def skip_log(self, msg: str):
        logging.warning("SKIP: " + msg)

    @abc.abstractmethod
    def fail_log(self, msg: str):
        logging.error("FAILE: " + msg)

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

    def run_test(self, quant=False, *args, **kwargs):
        status = True

        for prog_config in self.sample_program_configs(*args, **kwargs):
            # if program is invalid, we should skip that cases.
            if not self.is_program_valid(prog_config):
                continue

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
            logging.info('RUN program_config: ' + str(prog_config))
            results.append(
                self.run_test_config(model, params, prog_config, base_config,
                                     feed_data))
            self.success_log('RUN_CPU_BASELINE done')

            for paddlelite_config, (
                    atol, rtol) in self.sample_predictor_configs(prog_config):
                # skip info
                skip_flag = False
                pred_config = paddlelite_config.value()
                for skip_info in self.skip_cases:
                    if skip_info[0](prog_config, pred_config):
                        skip_flag = True
                        if skip_info[1] == SkipReasonsBase.ACCURACY_ERROR:
                            self.skip_log("[ACCURACY_ERROR] " +
                                          skip_info[2] + ' ' + ' vs ' + self.
                                          paddlelite_config_str(pred_config))
                        else:
                            raise NotImplementedError
                        break
                try:
                    results.append(self.run_lite_config(model, params, feed_data, pred_config))

                except Exception as e:
                    self.fail_log(
                        self.paddlelite_config_str(pred_config) +
                        '\033[1;31m \nERROR INFO: {}\033[0m'.format(str(e)))
                    if not skip_flag:
                        status = False
                    continue
                self.success_log('RUN predictor_config ' + self.
                                 paddlelite_config_str(pred_config) + ' done')

        self.assertTrue(status)

    def inference_config_str(self, config) -> bool:
        dic = {}
        enable_mkldnn = config.mkldnn_enabled()
        dic['use_mkldnn'] = enable_mkldnn
        enable_gpu = config.use_gpu()
        return str(dic)

    def paddlelite_config_str(self, config) -> bool:
        return str(config)

    @abc.abstractmethod
    def run_lite_config(self, model, params, feed_data, pred_config) -> Dict[str, np.ndarray]:
        raise NotImplementedError
