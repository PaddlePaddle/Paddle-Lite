# -*- coding: utf-8 -*-
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import argparse
import sys
import os
import re


def compute_vsdot_vec_vec(vd, vn, vm):
    i = 0x4e809400 | int(vd) | (int(vn) << 5) | (int(vm) << 16)
    return '".word 0x{:08x}\\n"'.format(i) + \
           ' /* vsdot v{vd}.4s, v{vn}.16b, v{vm}.16b */'.format(
               vd=vd, vn=vn, vm=vm)


def compute_vsdot_vec_elem(vd, vn, vm, idx):
    i = 0x0d40fe20 | ((int(vd) & 0x7) << 29) | ((int(vd) & 0x8) << 3) | (
        (int(vn) & 0x7) << 1) | (
            (int(vn) & 0x8) << 20) | (int(vm) << 16) | (int(idx % 2) << 21)
    return '".word 0x{:08x}\\n"'.format(i) + \
           ' /* vsdot.s8 q{vd}, q{vn}, d{vm}[{idx}] */\\\r\n'.format(
               vd=vd, vn=vn, vm=vm, idx=idx)


def match_vsdot_patten(line):
    matched = re.search(
        r'vsdot.s8\s+q(.*?)\s*,\s*q(.*?)\s*,\s*d(.*?)\[(.*?)\].*', line, re.M |
        re.I)
    if matched:
        vd = int(matched.group(1))
        vn = int(matched.group(2))
        vm = int(matched.group(3))
        idx = int(matched.group(4))
        return compute_vsdot_vec_elem(vd, vn, vm, idx)
    else:
        return line


def parser_file(file_in, file_out):
    out = open(file_out, 'w')
    if os.path.exists(file_in):
        for line in open(file_in):
            new_line = match_vsdot_patten(line)
            # print(new_line)
            out.write(new_line)
    else:
        print('input file {} not exist'.format(file_in))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser('convert arm vsdot to machine code')
    arg_parser.add_argument('--input_file', type=str, required=True)
    arg_parser.add_argument('--output_file', type=str, required=True)
    args = arg_parser.parse_args()

    print('input file: ', args.input_file)
    print('output file: ', args.output_file)
    parser_file(args.input_file, args.output_file)
