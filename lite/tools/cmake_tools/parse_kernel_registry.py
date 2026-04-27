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

if len(sys.argv) != 8:
    print("Error: parse_kernel_registry.py requires seven inputs!")
    exit(1)
kernels_list_path = sys.argv[1]
faked_kernels_list_path = sys.argv[2]
dest_header_path = sys.argv[3]
minkernels_list_path = sys.argv[4]
tailored = sys.argv[5]
with_extra = sys.argv[6]
enable_arm_fp16 = sys.argv[7]

# Generate .h path and .cc path
dest_cc_path = dest_header_path.replace('.h', '.cc')

header_lines = [
    '#pragma once',
    '#include "paddle_lite_factory_helper.h"',
    '',
]
minlines = set()
if tailored == "ON":
    with open(minkernels_list_path) as fd:
        for line in fd:
            minlines.add(line.strip())

# Collect all emitted kernels for RegisterAllKernels() below.
emitted_kernels = []

with open(kernels_list_path) as f:
    paths = set([path for path in f])
    for path in paths:
        with open(path.strip()) as g:
            c = g.read()
            kernel_parser = RegisterLiteKernelParser(c)
            kernel_parser.parse(with_extra, enable_arm_fp16)

            for k in kernel_parser.kernels:
                kernel = "%s,%s,%s,%s,%s" % (
                    k.op_type,
                    k.target,
                    k.precision,
                    k.data_layout,
                    k.alias, )
                if tailored == "ON":
                    if kernel not in minlines: continue
                key = "USE_LITE_KERNEL(%s, %s, %s, %s, %s);" % (
                    k.op_type,
                    k.target,
                    k.precision,
                    k.data_layout,
                    k.alias, )
                header_lines.append(key)
                emitted_kernels.append((k.op_type, k.target, k.precision,
                                        k.data_layout, k.alias))

with open(faked_kernels_list_path) as f:
    paths = set([path for path in f])
    for path in paths:
        if (sys.version[0] == '3'):
            with open(path.strip(), encoding='utf-8') as g:
                c = g.read()
                kernel_parser = RegisterLiteKernelParser(c)
                kernel_parser.parse(with_extra, "ON")

                for k in kernel_parser.kernels:
                    kernel = "%s,%s,%s,%s,%s" % (
                        k.op_type,
                        k.target,
                        k.precision,
                        k.data_layout,
                        k.alias, )
                    if tailored == "ON":
                        if kernel not in minlines: continue
                    key = "USE_LITE_KERNEL(%s, %s, %s, %s, %s);" % (
                        k.op_type,
                        k.target,
                        k.precision,
                        k.data_layout,
                        k.alias, )
                    header_lines.append(key)
                    emitted_kernels.append((k.op_type, k.target, k.precision,
                                            k.data_layout, k.alias))
        else:
            with open(path.strip()) as g:
                c = g.read()
                kernel_parser = RegisterLiteKernelParser(c)
                kernel_parser.parse(with_extra, "ON")

                for k in kernel_parser.kernels:
                    kernel = "%s,%s,%s,%s,%s" % (
                        k.op_type,
                        k.target,
                        k.precision,
                        k.data_layout,
                        k.alias, )
                    if tailored == "ON":
                        if kernel not in minlines: continue
                    key = "USE_LITE_KERNEL(%s, %s, %s, %s, %s);" % (
                        k.op_type,
                        k.target,
                        k.precision,
                        k.data_layout,
                        k.alias, )
                    header_lines.append(key)
                    emitted_kernels.append((k.op_type, k.target, k.precision,
                                            k.data_layout, k.alias))

# Append inline RegisterAllKernels() guarded by #ifdef LITE_LAZY_REGISTER.
# generate_factory_helper.py resolves this block at build time so the exported
# paddle_use_kernels.h contains the unconditional inline function when
# LITE_LAZY_REGISTER=ON, and nothing extra when LITE_LAZY_REGISTER=OFF.
header_lines.extend([
    '',
    '#ifdef LITE_LAZY_REGISTER',
    'namespace paddle {',
    'namespace lite {',
    'namespace detail {',
    '__attribute__((used)) inline void RegisterAllKernels() {',
])
for (op_type, target, precision, data_layout, alias) in emitted_kernels:
    func_name = 'touch_%s%s%s%s%s' % (op_type, target, precision, data_layout,
                                       alias)
    header_lines.append('  %s();' % func_name)
header_lines.extend([
    '}',
    '}  // namespace detail',
    '}  // namespace lite',
    '}  // namespace paddle',
    '#endif  // LITE_LAZY_REGISTER',
])

# Generate .h file
with open(dest_header_path, 'w') as f:
    logging.info("write kernel header to %s" % dest_header_path)
    f.write('\n'.join(header_lines))

# Generate .cc file: weak empty fallback compiled into the SDK.
# Overridden at link time by the inline RegisterAllKernels() from any consumer
# that #includes paddle_use_kernels.h (e.g. MMLNative).
cc_lines = [
    '// Auto-generated by parse_kernel_registry.py - DO NOT EDIT',
    '// SDK weak-empty fallback for RegisterAllKernels().',
    '// Overridden at link time by the inline version in paddle_use_kernels.h.',
    '',
    '#ifdef LITE_LAZY_REGISTER',
    'namespace paddle {',
    'namespace lite {',
    'namespace detail {',
    '__attribute__((weak)) void RegisterAllKernels() {}',
    '}  // namespace detail',
    '}  // namespace lite',
    '}  // namespace paddle',
    '#endif  // LITE_LAZY_REGISTER',
]

with open(dest_cc_path, 'w') as f:
    logging.info("write kernel source to %s" % dest_cc_path)
    f.write('\n'.join(cc_lines))
