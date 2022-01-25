# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
# this module will record kernels in unvalid_places into all_kernel_faked.cc
"""
Name: parse_tailored_kernel.py
Usage: to generate `xxx_compute_arm.cc`/`xxx_compute_host.cc` in build dir when tailor kernel.
"""
import sys
import os
from ast import RegisterLiteKernelParser


def parse_tailored_kernel_from_file(src_path, dst_path, op_name, device_target,
                                    data_type, layout_type, alias_name,
                                    first_flag):
    with open(src_path) as src:
        src_content = src.read()
        kernel_parser = RegisterLiteKernelParser(src_content)
        kernel_parser.pick_kernel_class(op_name, device_target, data_type,
                                        layout_type, alias_name, first_flag,
                                        dst_path)


def main(argv):
    if len(argv) != 9:
        print("Error: parse_tailored_kernel.py requires 8 inputs!")
        exit(1)
    src_kernel_file = argv[1]
    dst_kernel_path = argv[2]
    op_name = argv[3]
    device_target = argv[4]
    data_type = argv[5]
    layout_type = argv[6]
    alias_name = argv[7]
    first_flag = argv[8]
    file_path = os.path.realpath(__file__)
    target = device_target[1:].lower()
    src_kernel_path = os.path.dirname(
        file_path) + "/../../kernels/" + target + "/" + src_kernel_file
    if target == "metal":
        src_kernel_path = os.path.dirname(
            file_path
        ) + "/../../kernels/" + target + "/image_op/" + src_kernel_file
    parse_tailored_kernel_from_file(src_kernel_path, dst_kernel_path, op_name,
                                    device_target, data_type, layout_type,
                                    alias_name, first_flag)


if __name__ == '__main__':
    main(sys.argv)
