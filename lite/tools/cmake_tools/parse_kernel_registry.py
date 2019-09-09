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

ops_list_path = sys.argv[1]
dest_path = sys.argv[2]

out_lines = [
    '#pragma once',
    '#include "paddle_lite_factory_helper.h"',
    '',
]


with open(ops_list_path) as f:
    paths = set([path for path in f])
    for path in paths:
        with open(path.strip()) as g:
            print 'path: ', path
            c = g.read()
            kernel_parser = RegisterLiteKernelParser(c)
            kernel_parser.parse()

            for k in kernel_parser.kernels:
                  key = "USE_LITE_KERNEL(%s, %s, %s, %s, %s);" % (
                     k.op_type,
                     k.target,
                     k.precision,
                     k.data_layout,
                     k.alias,
                  )
                  out_lines.append(key)

with open(dest_path, 'w') as f:
    logging.info("write kernel list to %s" % dest_path)
    f.write('\n'.join(out_lines))
