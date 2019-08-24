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

        with open(path) as g:
            for line in g:
                key = 'REGISTER_LITE_OP'
                if line.startswith(key):
                    end = line.find(',')
                    op = line[len(key) + 1:end]
                    if not op: continue
                    if "_grad" in op: continue
                    out = "USE_LITE_OP(%s);" % op
                    out_lines.append(out)

with open(dest_path, 'w') as f:
    logging.info("write op list to %s" % dest_path)
    f.write('\n'.join(out_lines))
