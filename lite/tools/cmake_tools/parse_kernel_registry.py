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

ops_list_path = sys.argv[1]
dest_path = sys.argv[2]

out_lines = [
    '#pragma once',
    '#include "paddle_lite_factory_helper.h"',
    '',
]

left_pattern = 'REGISTER_LITE_KERNEL('
right_pattern = ')'

def find_right_pattern(context, start):
   if start >= len(context): return -1
   fake_left_num = 0
   while start < len(context):
       if context[start] == right_pattern:
           if fake_left_num == 0:
               return start
           else:
               fake_left_num -= 1
       elif context[start] == '(':
           fake_left_num += 1
       start += 1
   return -1

lines = set()
with open(ops_list_path) as f:
    for line in f:
        lines.add(line.strip())
    
for line in lines:
    path = line.strip()

    status = ''
    with open(path) as g:
        context = ''.join([item.strip() for item in g])
        index = 0
        cxt_len = len(context)
        while index < cxt_len and index >= 0:
            left_index = context.find(left_pattern, index)
            if left_index < 0: break
            right_index = find_right_pattern(context, left_index+len(left_pattern))
            if right_index < 0:
                raise ValueError("Left Pattern and Right Pattern does not match")
            tmp = context[left_index+len(left_pattern) : right_index]
            index = right_index + 1
            if tmp.startswith('/'): continue
            fields = [item.strip() for item in tmp.split(',')]
            if len(fields) < 6:
                raise ValueError("Invalid REGISTER_LITE_KERNEL format")

            op, target, precision, layout = fields[:4] 
            alias = fields[-1]
            key = "USE_LITE_KERNEL(%s, %s, %s, %s, %s);" % (
                op, target, precision, layout, alias)
            out_lines.append(key)


with open(dest_path, 'w') as f:
    logging.info("write kernel list to %s" % dest_path)
    f.write('\n'.join(out_lines))
