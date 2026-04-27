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
'''
Resolve #ifdef LITE_LAZY_REGISTER blocks in generated SDK headers.

Used for paddle_lite_factory_helper.h, paddle_use_kernels.h and
paddle_use_ops.h.  Replaces each #ifdef LITE_LAZY_REGISTER ... #endif
block with the appropriate branch based on build configuration, so that
APP developers don't need to define LITE_LAZY_REGISTER themselves.
'''

from __future__ import print_function
import sys
import re

if len(sys.argv) != 4:
    print("Error: generate_factory_helper.py requires three inputs!")
    print("Usage: generate_factory_helper.py <source_file> <dest_file> <LITE_LAZY_REGISTER>")
    exit(1)

source_path = sys.argv[1]
dest_path = sys.argv[2]
lite_lazy_register = sys.argv[3].upper() in ("ON", "TRUE", "1")

with open(source_path, 'r') as f:
    content = f.read()

# Pattern matches both forms:
#   #ifdef LITE_LAZY_REGISTER ... #else ... #endif  (group1=lazy, group2=else)
#   #ifdef LITE_LAZY_REGISTER ... #endif            (group1=lazy, group2=None)
# The optional trailing comment on #endif (e.g. "// LITE_LAZY_REGISTER") is consumed.
pattern = (r'#ifdef\s+LITE_LAZY_REGISTER\b(.*?)'
           r'(?:#else\b(.*?))?'
           r'#endif(?:[^\S\n]*//[^\n]*)?')

def replace_ifdef(match):
    """Replace #ifdef block based on LITE_LAZY_REGISTER value."""
    lazy_branch = match.group(1)
    else_branch = match.group(2) or ''
    if lite_lazy_register:
        return lazy_branch
    else:
        return else_branch

# Process the content
output = re.sub(pattern, replace_ifdef, content, flags=re.DOTALL)

with open(dest_path, 'w') as f:
    f.write(output)
