# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import json
from rpc_service.cxx_service.auto_scan_test_rpc_cxx_base import AutoScanCxxBaseTest
import numpy as np
import logging
import abc
import enum
import unittest
import paddle
import copy
from typing import Optional, List, Callable, Dict, Any, Set
import os
import subprocess


class ADBWrapper():
    def __init__(self, adb_device_name):
        self.adb_device_name = adb_device_name

    def push(self, src_path, dst_path):
        cmd = "adb -s {} push {} {}".format(self.adb_device_name, src_path,
                                            dst_path)
        p = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        logging.info(str(stdout, encoding='utf-8'))

    def pull(self, src_path, dst_path):
        cmd = "adb -s {} pull {} {}".format(self.adb_device_name, src_path,
                                            dst_path)
        p = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        logging.info(str(stdout, encoding='utf-8'))

    def run_command(self, cmd):
        cmd = "adb -s {} shell '{}'".format(self.adb_device_name, cmd)
        p = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        logging.info(str(stdout, encoding='utf-8'))


class AutoScanTest(AutoScanCxxBaseTest):
    def __init__(self, *args, **kwargs):
        super(AutoScanTest, self).__init__(*args, **kwargs)
        self.adb_device_name = self.args.adb_device_name
        self.adb_wrapper = ADBWrapper(self.adb_device_name)
        self.init()

    def check_params_valid(self):
        assert (self.run_mode == "adb")
        if self.adb_device_name == "":
            raise ValueError(
                "The adb_device_name is empty, please set the adb_device_name use --adb_device_name"
            )

        p = subprocess.Popen(
            "adb devices",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if self.adb_device_name not in str(stdout, encoding='utf-8'):
            raise ValueError(
                "The adb_device_name {} is invalid, the list of all devices is listed below:\n {}".
                format(
                    self.adb_device_name, str(stdout, encoding='utf-8')))

    def init(self):
        logging.info("ADB process init start ... ")
        super(AutoScanTest, self).init()
        self.check_params_valid()
        self.adb_wrapper.run_command(
            cmd="rm -rf {}/autoscan_cxx_test_tools".format(
                self.remote_work_dir))
        logging.info(
            "ADB wrapper upload autoscan cxx test demo from local to remote start ..."
        )
        self.adb_wrapper.push(
            src_path=self.model_test_demo_dir, dst_path=self.remote_work_dir)
        logging.info(
            "ADB wrapper upload autoscan cxx test demo from local to remote success."
        )
        logging.info("ADB process init done.")

    def run_cxx_test(self):
        # 1. Upload model/data to target machine
        test_info_src_dir = self.cache_dir
        test_info_dst_dir = self.remote_work_dir + "/autoscan_cxx_test_tools/test"
        self.adb_wrapper.run_command(
            cmd="rm -rf {}/autoscan_cxx_test_tools/test".format(
                self.remote_work_dir))
        self.adb_wrapper.run_command(
            cmd="mkdir -p {}/autoscan_cxx_test_tools/test".format(
                self.remote_work_dir))
        logging.info(
            "ADB wrapper upload test model and data from local to remote start ..."
        )
        self.adb_wrapper.push(
            src_path="{}/.".format(test_info_src_dir),
            dst_path=test_info_dst_dir)
        logging.info(
            "ADB wrapper upload test model and data from local to remote done")
        # 2. Run test demo
        logging.info("Run autoscan cxx test in remote ...")
        run_cmd = "cd {}/autoscan_cxx_test_tools; {} ./run.sh".format(
            self.remote_work_dir, self.remote_environment_variable)
        self.adb_wrapper.run_command(cmd=run_cmd)

    def get_result(self, prog_config):
        output_names = prog_config.outputs
        test_info_dst_dir = self.remote_work_dir + "/autoscan_cxx_test_tools/test"
        for output_name in output_names:
            output_tensor_file = "{}/{}.bin".format(test_info_dst_dir,
                                                    output_name)
            self.adb_wrapper.pull(
                src_path=output_tensor_file, dst_path=self.cache_dir)

        output_model_info_file = "{}/output.json".format(test_info_dst_dir)
        self.adb_wrapper.pull(
            src_path=output_model_info_file, dst_path=self.cache_dir)

        with open(self.cache_dir + "/output.json", "r") as f:
            output_json = json.load(f)

        result = {}
        for output_name in output_names:
            for elem in output_json['outputs']:
                if elem['name'] == output_name:
                    output_json_content = elem
            output_shape = output_json_content['shape']
            output_dtype = output_json_content['dtype']
            output_file = "{}/{}.bin".format(self.cache_dir, output_name)
            output_ = np.fromfile(
                output_file, dtype=output_dtype).reshape(output_shape)
            result[output_name] = output_

        result_res = copy.deepcopy(result)
        return result_res


class FusePassAutoScanTest(AutoScanTest):
    def run_and_statis(self,
                       quant=False,
                       max_examples=100,
                       reproduce=None,
                       min_success_num=25,
                       passes=None):
        assert passes is not None, "Parameter of passes must be defined in function run_and_statis."
        super().run_and_statis(quant, max_examples, reproduce, min_success_num,
                               passes)
