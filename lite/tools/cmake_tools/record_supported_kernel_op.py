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
# this module will record supported ops from kernels_src.txt

from __future__ import print_function
import sys
import os
import logging
from ast import RegisterLiteKernelParser
from ast import RegisterLiteOpParser
from ast import RegisterSubgraphBridgeParser
from ast import RegisterNNadapterBridgeParser

if len(sys.argv) != 6:
    print("Error: record_supported_kernel_op.py requires five inputs!")
    sys.exit(1)
kernels_list_path = sys.argv[1]
faked_kernels_list_path = sys.argv[2]
ops_list_path = sys.argv[3]
kernel_op_map_dest_path = sys.argv[4]
with_extra = sys.argv[5]

out_lines = [
    '''
// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include<vector>
#include<map>
#include<string>

const std::vector<std::vector<std::string>> supported_ops_target = {
'''
]

ops_lines = []

# valid targets and valid_ops
valid_targets = [
    "kUnk", "kHost", "kX86", "kCUDA", "kARM", "kOpenCL", "kAny", "kFPGA",
    "kNPU", "kXPU", "kBM", "kMLU", "kIntelFPGA", "kMetal", "kNNAdapter"
]
valid_ops = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
             [], [], []]


class TargetType:
    kUnk = 0
    kHost = 1
    kX86 = 2
    kCUDA = 3
    kARM = 4
    kOpenCL = 5
    kAny = 6  # any target
    kFPGA = 7
    kNPU = 8
    kXPU = 9
    kBM = 10
    kMLU = 11
    kRKNPU = 12
    kIntelFPGA = 16
    kMetal = 17
    kNNAdapter = 18


# record op_info of valid kernels into `valid_ops` according to different target type
with open(kernels_list_path) as f:
    paths = set([path for path in f])
    for path in paths:
        with open(path.strip()) as g:
            c = g.read()
            kernel_parser = RegisterLiteKernelParser(c)
            kernel_parser.parse("ON", "ON")
            for k in kernel_parser.kernels:
                if hasattr(TargetType, k.target):
                    index = getattr(TargetType, k.target)
                    valid_ops[index].append(k.op_type)
# record op_info of valid kernels into `valid_ops` according to different target type
with open(faked_kernels_list_path) as f:
    paths = set([path for path in f])
    for path in paths:
        if (sys.version[0] == '3'):
            with open(path.strip(), encoding='utf-8') as g:
                c = g.read()
                kernel_parser = RegisterLiteKernelParser(c)
                kernel_parser.parse("ON", "ON")
                for k in kernel_parser.kernels:
                    if hasattr(TargetType, k.target):
                        index = getattr(TargetType, k.target)
                        valid_ops[index].append(k.op_type)
        else:
            with open(path.strip()) as g:
                c = g.read()
                kernel_parser = RegisterLiteKernelParser(c)
                kernel_parser.parse("ON", "ON")
                for k in kernel_parser.kernels:
                    if hasattr(TargetType, k.target):
                        index = getattr(TargetType, k.target)
                        valid_ops[index].append(k.op_type)

# clear the repeated ops
for target in valid_targets:
    index = getattr(TargetType, target)
    valid_ops[index] = list(set(valid_ops[index]))

paths = set()
with open(ops_list_path) as f:
    paths = set([path for path in f])
    for path in paths:
        str_info = open(path.strip()).read()
        op_parser = RegisterLiteOpParser(str_info)
        ops = op_parser.parse(with_extra)
        for op in ops:
            if "_grad" in op:
                continue
            out = '    {"%s", { "' % op
            op_targets = []
            for target in valid_targets:
                if op in valid_ops[getattr(TargetType, target)]:
                    op_targets.append(target)
            if len(op_targets) > 0:
                out = out + '", "'.join(op_targets) + '" }}'
            else:
                # unknow type op:  kUnk = 0
                valid_ops[0].append(op)
                out = out + 'kUnk" }}'
            ops_lines.append(out)

with open(kernel_op_map_dest_path, 'w') as f:
    logging.info("write kernel list to %s" % kernel_op_map_dest_path)
    f.write('\n'.join(out_lines))
    # write kernels into head file
    for target in valid_targets:
        if len(valid_ops[getattr(TargetType, target)]) == 0:
            f.write("\n    // %s_OPS: " % target)
            f.write('\n    {},')
        else:
            f.write("\n    // %s_OPS: " % target)
            f.write('\n    {"')
            f.write('","'.join(valid_ops[getattr(TargetType, target)]))
            f.write('"},\n')
    f.write('};')
    # write op info into head file
    f.write(
        '\nconst std::map<std::string, std::vector<std::string>> supported_ops={\n'
    )
    f.write(',\n'.join(ops_lines))
    f.write('\n};')
