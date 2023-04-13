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
parser.add_argument("template_file")
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
    command_line = args[0]
    command_args = ""
    for i in range(1, len(args)):
        command_line = command_line + " " + args[i]
    for i in range(3, len(args)):
        command_args = command_args + " " + args[i]
    print(command_line)
    args = parser.parse_args()
    template_defines = dict(chain(*args.defines))
    template_defines["TEMPLATE_FILE"] = args.template_file
    template_defines["OUTPUT_FILE"] = args.output_file
    template_defines["COMMAND_LINE"] = command_line
    template_defines["COMMAND_ARGS"] = command_args
    template_text = codecs.open(
        args.template_file, "r", encoding="utf-8").read()
    template_lines = template_text.splitlines()
    python_lines = []
    blank_lines = 0
    last_line = ""
    last_indent = ""
    # List of tuples (total_index, python_indent)
    indent_stack = [("", "")]
    # Indicates whether this is the first line inside Python
    # code block (i.e. for, while, if, elif, else)
    python_block_start = True
    for i, template_line in enumerate(template_lines):
        if template_line == "":
            blank_lines += 1
            continue
        # Skip lint markers.
        #if 'LINT' in template_line:
        #  continue

        template_indent = extract_leading_whitespace(template_line)
        if python_block_start:
            assert template_indent.startswith(last_indent)
            extra_python_indent = template_indent[len(last_indent):]
            python_indent = indent_stack[-1][1] + extra_python_indent
            indent_stack.append((template_indent, python_indent))
            assert template_indent.startswith(indent_stack[-1][0])
        else:
            while not template_indent.startswith(indent_stack[-1][0]):
                del indent_stack[-1]
        python_block_start = False

        python_indent = indent_stack[-1][1]
        stripped_template_line = template_line.strip()
        if stripped_template_line.startswith(
                "$") and not stripped_template_line.startswith("${"):
            if stripped_template_line.endswith(":"):
                python_block_start = True
            while blank_lines != 0:
                python_lines.append(python_indent + "print(file=OUT_STREAM)")
                blank_lines -= 1
            python_lines.append(python_indent + stripped_template_line.replace(
                "$", ""))
        else:
            assert template_line.startswith(python_indent)
            while blank_lines != 0:
                python_lines.append(python_indent + "print(file=OUT_STREAM)")
                blank_lines -= 1
            python_lines.append(python_indent + "print(%s, file=OUT_STREAM)" %
                                escape(template_line[len(python_indent):]))
        last_line = template_line
        last_indent = template_indent

    while blank_lines != 0:
        python_lines.append(python_indent + "print(file=OUT_STREAM)")
        blank_lines -= 1

    exec_globals = dict(template_defines)
    if sys.version_info > (3, 0):
        output_stream = io.StringIO()
    else:
        output_stream = io.BytesIO()
    exec_globals["OUT_STREAM"] = output_stream
    python_bytecode = compile("\n".join(python_lines), args.template_file,
                              'exec')
    exec(python_bytecode, exec_globals)
    with codecs.open(args.output_file, "w", encoding="utf-8") as output_file:
        output_file.write(output_stream.getvalue())


if __name__ == "__main__":
    main(sys.argv)
