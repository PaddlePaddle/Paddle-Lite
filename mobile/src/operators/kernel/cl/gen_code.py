#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import re
import os
import sys

source = """
#pragma
#ifdef PADDLE_MOBILE_CL
#include <map>
#include <string>
#include <vector>
namespace paddle_mobile {
    extern const std::map<std::string, std::vector<unsigned char>> opencl_kernels = {
%s
    };
    extern const std::vector<std::string> need_conv_header_kernels = {
        %s
    };
}
#endif
"""

def string_to_hex(str):
    hex_list = []
    for i in range(len(code_str)):
        hex_ = hex(ord(code_str[i]))
        hex_list.append(hex_)
    return hex_list

infile = open("cl_kernel/cl_common.h", "r")
common_content = infile.read()
infile.close()
common_content = re.sub(r"/\*[^*]*\*/", "", common_content, flags=re.DOTALL)
lines = common_content.split("\n")
new_lines = []
for i in range(len(lines)):
    line = lines[i]
    line = line.strip()
    if line == "":
        continue
    if line.startswith("//"):
        continue
    line = re.sub(r"//.*$", "", line)
    new_lines.append(line)
common_content = "\n".join(new_lines)

need_conv_header_kernels = []

cores = ""
filenames = os.listdir("cl_kernel")
file_count = len(filenames)
for i in range(file_count):
    filename = filenames[i]
    infile = open("cl_kernel/" + filename, "r")
    new_lines = []
    content = infile.read()
    content = re.sub(r"/\*[^*]*\*/", "", content, flags=re.DOTALL)
    infile.close()
    lines = content.split("\n")
    for i in range(len(lines)):
        line = lines[i]
        line = line.strip()
        if line == "":
            continue
        if line.startswith("//"):
            continue
        line = re.sub(r"//.*$", "", line)
        if "cl_common.h" in line:
            line = common_content
        elif "conv_kernel.inc.cl" in line:
            need_conv_header_kernels.append("\"%s\"" % filename)
            continue
        new_lines.append(line)
    content = "\n".join(new_lines)
    if content == "":
        content = " "
    hexes = []
    for char in content:
        hexes.append(hex(ord(char)))
    core = "        {\"%s\", {" % filename
    for item in hexes:
        core += str(item) + ", "
    core = core[: -2]
    core += "}}"
    if i != file_count - 1:
        core += ",\n"
    cores += core

source = source % (cores, ",".join(need_conv_header_kernels))
print(source)
