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
''' Collect op registry information. '''

from __future__ import print_function
import sys
import logging
from ast import RegisterLiteOpParser

if len(sys.argv) != 6:
    print("Error: parse_op_registry.py requires four inputs!")
    exit(1)
ops_list_path = sys.argv[1]
dest_path = sys.argv[2]
minops_list_path = sys.argv[3]
tailored = sys.argv[4]
with_extra = sys.argv[5]
out_lines = [
    '#pragma once',
    '#include "paddle_lite_factory_helper.h"',
    '',
]

paths = set()
for line in open(ops_list_path):
    paths.add(line.strip())

if tailored == "ON":
    minlines = set()
    with open(minops_list_path) as fd:
        for line in fd:
            minlines.add(line.strip())
for path in paths:
    str_info = open(path.strip()).read()
    op_parser = RegisterLiteOpParser(str_info)
    ops = op_parser.parse(with_extra)
    for op in ops:
        if tailored == "ON":
            if op not in minlines: continue
        out = "USE_LITE_OP(%s);" % op
        out_lines.append(out)

with open(dest_path, 'w') as f:
    logging.info("write op list to %s" % dest_path)
    f.write('\n'.join(out_lines))
