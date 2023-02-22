#!/usr/bin/env python
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import codecs
import io
import re
import sys
from itertools import chain


def key_value(argument):
    key, value = argument.split("=", 1)
    try:
        value = int(value)
    except ValueError:
        pass
    return key, value


parser = argparse.ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("output_file")
parser.add_argument(
    "-D", dest="defines", nargs="*", type=key_value, action="append")
parser.set_defaults(defines=list())


def extract_leading_whitespace(line):
    match = re.match(r"\s*", line)
    return match.group(0) if match else ""


def escape(line):
    output_parts = []
    while "${" in line:
        start_pos = line.index("${")
        end_pos = line.index("}", start_pos + 2)
        if start_pos != 0:
            output_parts.append("\"" + line[:start_pos].replace("\"", "\\\"") +
                                "\"")
        output_parts.append("str(" + line[start_pos + 2:end_pos] + ")")
        line = line[end_pos + 1:]
    if line:
        output_parts.append("\"" + line.replace("\"", "\\\"") + "\"")
    return " + ".join(output_parts)


def main(args):
    args = parser.parse_args()
    print(args)
    input_defines = dict(chain(*args.defines))
    input_text = codecs.open(args.input_file, "r", encoding="utf-8").read()
    input_lines = input_text.splitlines()
    python_lines = []
    blank_lines = 0
    last_line = ""
    last_indent = ""
    # List of tuples (total_index, python_indent)
    indent_stack = [("", "")]
    # Indicates whether this is the first line inside Python
    # code block (i.e. for, while, if, elif, else)
    python_block_start = True
    for i, input_line in enumerate(input_lines):
        if input_line == "":
            blank_lines += 1
            continue
        # Skip lint markers.
        #if 'LINT' in input_line:
        #  continue

        input_indent = extract_leading_whitespace(input_line)
        if python_block_start:
            assert input_indent.startswith(last_indent)
            extra_python_indent = input_indent[len(last_indent):]
            python_indent = indent_stack[-1][1] + extra_python_indent
            indent_stack.append((input_indent, python_indent))
            assert input_indent.startswith(indent_stack[-1][0])
        else:
            while not input_indent.startswith(indent_stack[-1][0]):
                del indent_stack[-1]
        python_block_start = False

        python_indent = indent_stack[-1][1]
        stripped_input_line = input_line.strip()
        if stripped_input_line.startswith(
                "$") and not stripped_input_line.startswith("${"):
            if stripped_input_line.endswith(":"):
                python_block_start = True
            while blank_lines != 0:
                python_lines.append(python_indent + "print(file=OUT_STREAM)")
                blank_lines -= 1
            python_lines.append(python_indent + stripped_input_line.replace(
                "$", ""))
        else:
            assert input_line.startswith(python_indent)
            while blank_lines != 0:
                python_lines.append(python_indent + "print(file=OUT_STREAM)")
                blank_lines -= 1
            python_lines.append(python_indent + "print(%s, file=OUT_STREAM)" %
                                escape(input_line[len(python_indent):]))
        last_line = input_line
        last_indent = input_indent

    while blank_lines != 0:
        python_lines.append(python_indent + "print(file=OUT_STREAM)")
        blank_lines -= 1

    exec_globals = dict(input_defines)
    if sys.version_info > (3, 0):
        output_stream = io.StringIO()
    else:
        output_stream = io.BytesIO()
    exec_globals["OUT_STREAM"] = output_stream
    python_bytecode = compile("\n".join(python_lines), args.input_file, 'exec')
    exec(python_bytecode, exec_globals)
    with codecs.open(args.output_file, "w", encoding="utf-8") as output_file:
        output_file.write("// Auto-generated file from " + args.input_file +
                          ", Don't edit it!\n" + output_stream.getvalue())


if __name__ == "__main__":
    main(sys.argv)
