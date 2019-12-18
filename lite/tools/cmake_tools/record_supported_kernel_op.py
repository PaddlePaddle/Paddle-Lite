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

import sys
import logging
from ast import RegisterLiteKernelParser
from ast import RegisterLiteOpParser

kernels_list_path = sys.argv[1]
ops_list_path = sys.argv[2]
kernel_op_map_dest_path = sys.argv[3] #"./supported_kernel_op_info.h"


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
// ATTENTION This can only include in a .cc file.

const std::vector<std::vector<std::string>> supported_ops_target = {
'''
]

ops_lines=[]

# valid targets and valid_ops
valid_ops=[[],[],[],[],[],[],[],[],[],[]]
valid_targets=["kUnk","kHost","kX86","kCUDA","kARM","kOpenCL","kFPGA","kNPU","kXPU","kAny"]
class TargetType:
    kUnk = 0
    kHost = 1
    kX86 = 2
    kCUDA = 3
    kARM = 4
    kOpenCL = 5
    kFPGA = 7
    kNPU = 8
    kXPU = 9
    kAny = 6  # any target


with open(kernels_list_path) as f:
    paths = set([path for path in f])
    for path in paths:
        with open(path.strip()) as g:
            c = g.read()
            kernel_parser = RegisterLiteKernelParser(c)
            kernel_parser.parse()
            for k in kernel_parser.kernels:
                  if hasattr(TargetType, k.target):
                      index=getattr(TargetType, k.target)
                      op_type_str = '"%s"' % k.op_type                    
                      valid_ops[index].append(k.op_type)

paths = set()
for line in open(ops_list_path):
    paths.add(line.strip())
for path in paths:
    str_info = open(path.strip()).read()
    op_parser = RegisterLiteOpParser(str_info)
    ops = op_parser.parse()
    for op in ops:
        if "_grad" in op: 
            continue
        out = '    {"%s", { "' % op
        op_targets = []
        for target in valid_targets:
           if(op in valid_ops[getattr(TargetType, target)]):
              op_targets.append(target)
        out = out +'", "'.join(op_targets)+ '" }}'
        ops_lines.append(out)

with open(kernel_op_map_dest_path, 'w') as f:
    logging.info("write kernel list to %s" % kernel_op_map_dest_path)
    f.write('\n'.join(out_lines))
    # write kernels into head file
    for target in valid_targets:
       f.write("\n    // %s_OPS: " %target)
       f.write('\n    {"')
       f.write('","'.join(valid_ops[getattr(TargetType, target)]))
       f.write('"},\n')
    f.write('};')
    # write op info into head file
    f.write('\nconst std::map<std::string, std::vector<std::string>> supported_ops={\n')
    f.write(',\n'.join(ops_lines))
    f.write('\n};')
