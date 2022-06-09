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
from auto_scan_base import AutoScanBaseTest
import numpy as np
import logging
import abc
import enum
import unittest
import paddle
import copy
from typing import Optional, List, Callable, Dict, Any, Set
import os
import sys
import paramiko
from scp import SCPClient
import traceback
import subprocess


class AutoScanCxxBaseTest(AutoScanBaseTest):
    def __init__(self, *args, **kwargs):
        super(AutoScanCxxBaseTest, self).__init__(*args, **kwargs)
        self.remote_work_dir = self.args.remote_work_dir
        self.run_mode = self.args.run_mode
        self.target_os = self.args.target_os
        self.target_arch = self.args.target_arch
        self.target_abi = self.args.target_abi
        self.remote_env_variable = self.args.remote_env_variable
        self.host_android_ndk_path = self.args.host_android_ndk_path
        self.model_test_demo_dir = os.path.abspath(os.path.dirname(
            __file__)) + "/autoscan_cxx_test_tools"
        self.remote_environment_variable = ""
        self.check_base_param_valid()
        self.download_cxx_test_demo()

    def download_cxx_test_demo(self):
        demo_url = "http://paddlelite-data.bj.bcebos.com/autoscan/autoscan_cxx_test_tools.tar.gz"
        if not os.path.exists(self.model_test_demo_dir):
            cmd = "curl {} -o -| tar -xz -C {} ".format(
                demo_url, os.path.abspath(os.path.dirname(__file__)))
            os.system(cmd)
        if not os.path.exists(self.model_test_demo_dir):
            raise ValueError(
                "Download cxx test demo failed from url: {}!".format(demo_url))

    @abc.abstractmethod
    def check_params_valid(self):
        raise NotImplementedError

    @abc.abstractmethod
    def init(self):
        # 1. Package paddlelite libs from build dir to test demo
        self.package_paddlelite_libs_from_build_dir()
        # 2. According to the target_os/target_abi build test demo
        self.build_test_demo()
        # 3. Init remote environment variable
        env_list = self.remote_env_variable.split(";")
        variable_list = []
        for key_value in env_list:
            if len(key_value) < 1:
                continue
            variable_list.append(key_value)
        if len(variable_list) != 0:
            if not variable_list[-1].endswith(";"):
                variable_list[-1] = variable_list[-1] + ";"
            self.remote_environment_variable = ";".join(variable_list)

    def check_base_param_valid(self):
        if self.remote_work_dir == "":
            raise ValueError(
                "The remote work dir is empty, please set the remote work dir use --remote_work_dir"
            )
        if self.target_os == "":
            raise ValueError(
                "The target os is empty, please set the target os use --target_os"
            )
        if self.target_arch == "":
            raise ValueError(
                "The target arch is empty, please set the target arch use --target_arch"
            )
        if self.target_abi == "":
            raise ValueError(
                "The target abi is empty, please set the target abi use --target_abi"
            )
        if self.target_os == "android":
            if self.host_android_ndk_path == "":
                raise ValueError(
                    "If target_os is android, need set host_android_ndk_path use --host_android_ndk_path"
                )
            if self.target_abi not in ['arm64-v8a', 'armeabi-v7a']:
                raise ValueError(
                    "If target_os is android, need set target_abi use --target_abi, choices=['arm64-v8a','armeabi-v7a']"
                )
        elif self.target_os == "linux":
            if self.target_abi not in ['arm64', 'armhf', 'amd64']:
                raise ValueError(
                    "If target_os is linux, need set target_abi use --target_abi, choices=['arm64','armhf','amd64']"
                )
        else:
            raise ValueError(
                "rpc cxx process not support the target os: {}, only support andorid or linux now".
                format(self.target_os))

    def build_test_demo(self):
        cmd = "./build.sh {} {} {}".format(self.target_os, self.target_abi,
                                           self.host_android_ndk_path)
        try:
            subprocess.check_call(
                cmd, shell=True, cwd=self.model_test_demo_dir)
        except subprocess.CalledProcessError as e:
            logging.fatal("cmd:{}".format(e.cmd))
            logging.fatal("output:{}".format(e.output))
            logging.fatal("returncode:{}".format(e.returncode))
            exit()

    def package_paddlelite_libs_from_build_dir(self):
        root_dir = "{}/../../../../..".format(
            os.path.abspath(os.path.dirname(__file__)))
        build_dir_prefix = "build.lite.{}.{}".format(self.target_os,
                                                     self.target_arch)
        build_dir = ""
        for name in os.listdir(root_dir):
            if name.startswith(build_dir_prefix):
                build_dir = os.path.join(root_dir, name)
                break
        if build_dir == "":
            raise ValueError(
                "Cant't find build.lite.{}.{}.* in root dir: {}, please build first!".
                format(self.target_os, self.target_arch, root_dir))

        inference_lib_dir = ""
        for dir_name in os.listdir(build_dir):
            if dir_name.startswith("inference_lite_lib"):
                inference_lib_dir = os.path.join(build_dir, dir_name)
                break
        if inference_lib_dir == "":
            raise ValueError("Cant't find inference_lite_lib.* in {}".format(
                build_dir))
        cxx_include_path = "{}/cxx/include".format(inference_lib_dir)
        cxx_lib_path = "{}/cxx/lib".format(inference_lib_dir)
        demo_lib_path = "{}/libs/PaddleLite".format(self.model_test_demo_dir)
        if os.path.exists(demo_lib_path):
            os.system("rm -rf {}".format(demo_lib_path))
        os.system("mkdir -p {}".format(demo_lib_path))
        os.system("cp -r {} {}".format(cxx_include_path, demo_lib_path))
        os.system("cp -r {} {}".format(cxx_lib_path, demo_lib_path))

    @abc.abstractmethod
    def run_cxx_test(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_result(self, prog_config):
        raise NotImplementedError

    def run_lite_config(self,
                        model,
                        params,
                        inputs,
                        pred_config,
                        prog_config,
                        server_ip="localhost") -> Dict[str, np.ndarray]:
        with open(self.cache_dir + "/model", "wb") as f:
            f.write(model)
        with open(self.cache_dir + "/params", "wb") as f:
            f.write(params)

        with open(self.cache_dir + "/model_info.json", "w") as f:
            model_info = {"inputs": []}
            for idx, name in enumerate(inputs):
                tensor_info = {
                    "name": name,
                    "shape": inputs[name]['data'].shape,
                    "dtype": str(inputs[name]['data'].dtype)
                }
                if inputs[name]['lod'] is not None:
                    tensor_info['lod'] = inputs[name]['lod']
                model_info["inputs"].append(tensor_info)
                inputs[name]['data'].tofile("{}/{}.bin".format(self.cache_dir,
                                                               name))
            model_info["configs"] = pred_config
            model_info["model_file"] = "model"
            model_info["param_file"] = "params"
            model_info["source_dir"] = "test"
            json.dump(model_info, f, indent=1)
        # Run cxx test
        self.run_cxx_test()

        # Collect result from remote
        result = self.get_result(prog_config)

        return result, None
