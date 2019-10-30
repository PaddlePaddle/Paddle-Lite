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

from __future__ import print_function
import sys
import logging
from ast import RegisterLiteKernelParser
from utils import *

ops_list_path = sys.argv[1]
dest_path = sys.argv[2]
kernelmap_path = sys.argv[3]

out_lines = [
    '#pragma once',
    '#include "lite/core/op_registry.h"',
    '#include "lite/core/kernel.h"',
    '#include "lite/core/type_system.h"',
    '',
]

fake_kernel = '''

namespace paddle {
namespace lite {

class %s : public KernelLite<TARGET(%s), PRECISION(%s), DATALAYOUT(%s)> {
 public:
  void PrepareForRun() override {}

  void Run() override {}

  virtual ~%s() = default;
};

}  // namespace lite
}  // namespace paddle
'''

# create .h file to store kernel&source relationship
kernel_src_map_lines = [
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
#include<map>
// ATTENTION This can only include in a .cc file.

const std::map<std::string, std::string> kernel2path_map{

'''
]


with open(ops_list_path) as f:
    paths = set([path for path in f])
    for path in paths:
        print('path', path)
        with open(path.strip()) as g:
            c = g.read()
            kernel_parser = RegisterLiteKernelParser(c)
            kernel_parser.parse()

            for k in kernel_parser.kernels:
                kernel_name = "{op_type}_{target}_{precision}_{data_layout}_{alias}_class".format(
                    op_type = k.op_type,
                    target = k.target,
                    precision = k.precision,
                    data_layout = k.data_layout,
                    alias = k.alias,
                )

                kernel_define = fake_kernel % (
                    kernel_name,
                    k.target,
                    k.precision,
                    k.data_layout,
                    kernel_name,
                )

                out_lines.append(kernel_define)
                out_lines.append("")


                key = "REGISTER_LITE_KERNEL(%s, %s, %s, %s, %s, %s)" % (
                    k.op_type,
                    k.target,
                    k.precision,
                    k.data_layout,
                    '::paddle::lite::' + kernel_name,
                    k.alias,
                )
                out_lines.append(key)

                for input in k.inputs:
                    io = '    .BindInput("%s", {%s})' % (input.name, input.type)
                    out_lines.append(io)
                for output in k.outputs:
                    io = '    .BindOutput("%s", {%s})' % (output.name, output.type)
                    out_lines.append(io)
                out_lines.append("    .Finalize();")
                out_lines.append("")
                out_lines.append(gen_use_kernel_statement(k.op_type, k.target, k.precision, k.data_layout, k.alias))

                index = path.rindex('/')
                filename = path[index + 1:]
                map_element = '  {"%s,%s,%s,%s,%s", "%s"},' % (
                    k.op_type,
                    k.target,
                    k.precision,
                    k.data_layout,
                    k.alias,
                    filename.strip()
                )
                kernel_src_map_lines.append(map_element)
with open(dest_path, 'w') as f:
    logging.info("write kernel list to %s" % dest_path)
    f.write('\n'.join(out_lines))

with open(kernelmap_path, 'w') as fd:
    logging.info("write kernel map to %s" % dest_path)
    kernel_src_map_lines.append('  {"  ", "  "}')
    kernel_src_map_lines.append('};')
    fd.write('\n'.join(kernel_src_map_lines))
