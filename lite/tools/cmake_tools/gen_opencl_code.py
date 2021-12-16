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
import logging

opencl_kernel_path = ""
opencl_dest_path = ""


def gen_opencl_kernels():
    source = """
#pragma
#ifdef LITE_WITH_OPENCL
#include <map>
#include <string>
#include <vector>
namespace paddle {
namespace lite {
    // file name => source
    extern const std::map<std::string, std::vector<unsigned char>> opencl_kernels_files = {
    %s
    };
} // namespace lite
} // namespace paddle
#endif
    """

    def clean_source(content):
        new_content = re.sub(r"/\*([^*]|[\r\n]|(\*+([^*/]|[\r\n])))*\*+/",
                             "",
                             content,
                             flags=re.DOTALL)
        lines = new_content.split("\n")
        new_lines = []
        for i in range(len(lines)):
            line = lines[i]
            line = re.sub(r"//.*$", "", line)
            line = line.strip()
            if line == "":
                continue
            new_lines.append(line)
        new_content = "\n".join(new_lines)
        return new_content

    infile = open(opencl_kernel_path + "/cl_common.h", "r")
    common_content = infile.read()
    infile.close()
    common_content = clean_source(common_content)

    def get_header_raw(content):
        lines = content.split("\n")
        new_lines = []
        for line in lines:
            if "__kernel void" in line:
                break
            new_lines.append(line)
        header = "\n".join(new_lines)
        return header

    common_header = get_header_raw(common_content)

    def get_header(content):
        lines = content.split("\n")
        new_lines = []
        for line in lines:
            if "__kernel" in line:
                break
            new_lines.append(line)
        for i in range(len(lines)):
            if "#include \"cl_common.h\"" in lines[
                    i] or "#include <cl_common.h>" in lines[i]:
                lines[i] = common_header
        header = "\n".join(lines)
        return header

    filenames = os.listdir(opencl_kernel_path + "/buffer")
    file_count = len(filenames)

    headers = {}
    funcs = {}
    for i in range(file_count):
        filename = filenames[i]
        infile = open(opencl_kernel_path + "/buffer/" + filename, "r")
        content = infile.read()
        infile.close()
        content = clean_source(content)
        header = get_header(content)
        headers["buffer/" + filename] = header

    image_filenames = os.listdir(opencl_kernel_path + "/image")
    image_file_count = len(image_filenames)

    for i in range(image_file_count):
        filename = image_filenames[i]
        infile = open(opencl_kernel_path + "/image/" + filename, "r")
        content = infile.read()
        infile.close()
        content = clean_source(content)
        header = get_header(content)
        headers["image/" + filename] = header

    core1 = ""
    for i in range(len(headers)):
        file_name = list(headers.keys())[i]
        content = headers[file_name]
        if content == "":
            content = " "
        hexes = []
        for char in content:
            hexes.append(hex(ord(char)))
        core = "        {\"%s\", {" % file_name
        for item in hexes:
            core += str(item) + ", "
        core = core[:-2]
        core += "}}"
        if i != len(headers) - 1:
            core += ",\n"
        core1 += core
    source = source % (core1)
    with open(opencl_dest_path, 'w') as f:
        logging.info("write opencl kernels source files to %s" %
                     opencl_dest_path)
        f.write(source)


def gen_empty_opencl_kernels():
    source = """
    #pragma once
    #ifdef PADDLE_MOBILE_CL
    #include <map>
    #include <string>
    #include <vector>
    namespace paddle_mobile {
        // func name => source
        extern const std::map<std::string, std::vector<unsigned char>> opencl_kernels = {
        };
    }
    #endif
    """


if __name__ == "__main__":
    opencl_kernel_path = sys.argv[1]
    opencl_dest_path = sys.argv[2]
    gen_opencl_kernels()
