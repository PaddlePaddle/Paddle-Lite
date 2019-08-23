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

with open(ops_list_path) as f:
    for line in f:
        path = line.strip()

        status = ''
        with open(path) as g:
            lines = [v for v in g]
            for i in range(len(lines)):
                line = lines[i].strip()

                if not status:
                    key = 'REGISTER_LITE_KERNEL'
                    if line.startswith(key):
                        forward = i + min(7, len(lines) - i)
                        remaining = line[len(key) + 1:] + ' '.join(
                            [v.strip() for v in lines[i + 1:forward]])

                        x = remaining.find('.')
                        if x > 0:
                            remaining = remaining[:x]

                        fs = [v.strip() for v in remaining.split(',')]
                        assert (len(fs) >= 4)
                        op, target, precision, layout, __, alias = fs[:6]
                        alias = alias.replace(')', '')

                        key = "USE_LITE_KERNEL(%s, %s, %s, %s, %s);" % (
                            op, target, precision, layout, alias)
                        out_lines.append(key)

with open(dest_path, 'w') as f:
    logging.info("write kernel list to %s" % dest_path)
    f.write('\n'.join(out_lines))
