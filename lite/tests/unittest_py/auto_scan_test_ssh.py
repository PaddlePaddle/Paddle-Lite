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
import sys
import paramiko
from scp import SCPClient
import traceback
import subprocess


class SshRemoteHost(object):
    def __init__(self, hostname, username, password, port=""):
        self.hostname_ = hostname
        self.username_ = username
        self.password_ = password
        self.port_ = port
        self.ssh_client_ = self.create_ssh_client()

    def create_ssh_client(self):
        '''
            Note: Use paramiko creates the ssh client. 
        '''
        ssh_client = paramiko.SSHClient()
        #自动添加策略，保存服务器的主机名和密钥信息
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(
            hostname=self.hostname_,
            username=self.username_,
            password=self.password_)
        return ssh_client

    def transfer_files(self,
                       local_path,
                       remote_path,
                       upload=True,
                       recursive=True):
        '''
            Note: Support file upload and download.
        '''

        def progress(filename, size, sent):
            # Define progress callback that prints the current percentage completed for the filed
            # print("INFO: scp {}'s progress: {:.2f}% ".format(filename, float(sent) / float(size) * 100))
            pass

        scp = SCPClient(
            self.ssh_client_.get_transport(),
            progress=progress,
            socket_timeout=100.0)
        try:
            if upload:
                scp.put(local_path, remote_path, recursive=recursive)
            else:
                scp.get(remote_path, local_path)
        except Exception as e:
            traceback.print_exc()
            return False
        finally:
            scp.close()
        return True

    def __del__(self):
        self.ssh_client_.close()

    def run_command(self, cmd, bufsize=-1, timeout=None, environment=None):
        stdin, stdout, stderr = self.ssh_client_.exec_command(
            command=cmd,
            bufsize=bufsize,
            timeout=timeout,
            environment=environment)
        logging.info(str(stdout.read(), encoding='utf-8'))
        return stdin, stdout, stderr


class AutoScanTest(AutoScanCxxBaseTest):
    def __init__(self, *args, **kwargs):
        super(AutoScanTest, self).__init__(*args, **kwargs)
        self.username = self.args.username
        self.ip = self.args.ip
        self.password = self.args.password
        self.ssh_client_ = None
        self.init()

    def init(self):
        logging.info("SSH process init start ... ")
        super(AutoScanTest, self).init()
        self.check_params_valid()
        self.ssh_client_ = SshRemoteHost(self.ip, self.username, self.password)
        self.ssh_client_.run_command(
            cmd="rm -rf {}/autoscan_cxx_test_tools".format(
                self.remote_work_dir))
        logging.info(
            "SSH client upload autoscan cxx test demo from local to remote start ..."
        )
        run_status = self.ssh_client_.transfer_files(
            local_path=self.model_test_demo_dir,
            remote_path=self.remote_work_dir,
            upload=True)
        if run_status == True:
            logging.info(
                "SSH client upload autoscan cxx test demo from local to remote succeed!"
            )
        else:
            raise ValueError(
                "SSH client upload autoscan cxx test demo from local to remote failed!"
            )
        logging.info("SSH process init done.")

    def check_params_valid(self):
        assert (self.run_mode == "ssh")
        if self.ip == "":
            raise ValueError(
                "The remote IP address is empty, please set the IP address use --ip"
            )
        if self.username == "":
            raise ValueError(
                "The remote username is empty, please set the username use --username"
            )
        if self.password == "":
            raise ValueError(
                "The remote password is empty, please set the password use --password"
            )

    def run_cxx_test(self):
        # 1. Upload model/data to target machine
        test_info_src_dir = self.cache_dir
        test_info_dst_dir = self.remote_work_dir + "/autoscan_cxx_test_tools/test"
        self.ssh_client_.run_command(
            cmd="rm -rf {}/autoscan_cxx_test_tools/test".format(
                self.remote_work_dir))
        self.ssh_client_.run_command(
            cmd="mkdir -p {}/autoscan_cxx_test_tools/test".format(
                self.remote_work_dir))
        logging.info(
            "SSH client upload test model and data from local to remote start ..."
        )
        for name in os.listdir(test_info_src_dir):
            file_path = os.path.join(test_info_src_dir, name)
            run_status = self.ssh_client_.transfer_files(
                local_path=file_path,
                remote_path=test_info_dst_dir,
                upload=True,
                recursive=False)
            if run_status == True:
                logging.info(
                    "SSH client upload {} from {} to remote {} succeed!".
                    format(name, file_path, test_info_dst_dir))
            else:
                raise ValueError(
                    "SSH client upload {} from {} to remote {} failed!".format(
                        name, file_path, test_info_dst_dir))
        logging.info(
            "SSH client upload test model and data from local to remote done")
        # 2. Run test demo
        logging.info("Run autoscan cxx test in remote ...")
        run_cmd = "cd {}/autoscan_cxx_test_tools; {} ./run.sh".format(
            self.remote_work_dir, self.remote_environment_variable)
        self.ssh_client_.run_command(cmd=run_cmd)

    def get_result(self, prog_config):
        output_names = prog_config.outputs
        test_info_dst_dir = self.remote_work_dir + "/autoscan_cxx_test_tools/test"
        for output_name in output_names:
            output_tensor_file = "{}/{}.bin".format(test_info_dst_dir,
                                                    output_name)
            run_status = self.ssh_client_.transfer_files(
                local_path=self.cache_dir,
                remote_path=output_tensor_file,
                upload=False)
            if run_status == True:
                logging.info(
                    "SSH client gets output tensor file {} success from remote!".
                    format(output_tensor_file))
            else:
                raise ValueError(
                    "SSH client gets output tensor file {} failed from remote!".
                    format(output_tensor_file))

        output_model_info_file = "{}/output.json".format(test_info_dst_dir)
        run_status = self.ssh_client_.transfer_files(
            local_path=self.cache_dir,
            remote_path=output_model_info_file,
            upload=False)
        if run_status == True:
            logging.info(
                "SSH client gets output model json file {} success from remote!".
                format(output_model_info_file))
        else:
            raise ValueError(
                "SSH client gets output model json file {} failed from remote!".
                format(output_model_info_file))

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
