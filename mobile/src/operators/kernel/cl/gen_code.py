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

def gen_opencl_kernels():
    source = """
    #pragma
    #ifdef PADDLE_MOBILE_CL
    #include <map>
    #include <string>
    #include <vector>
    namespace paddle_mobile {
        // func name => source
        extern const std::map<std::string, std::vector<unsigned char>> opencl_kernels = {
    %s
        };
        // file name => header
        extern const std::map<std::string, std::vector<unsigned char>> opencl_headers = {
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

    def clean_source(content):
        new_content = re.sub(r"/\*([^*]|[\r\n]|(\*+([^*/]|[\r\n])))*\*+/", "", content, flags=re.DOTALL)
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

    infile = open("cl_kernel/cl_common.h", "r")
    common_content = infile.read()
    infile.close()
    common_content = clean_source(common_content)

    infile = open("cl_kernel/conv_kernel.inc.cl", "r")
    inc_content = infile.read()
    infile.close()
    inc_content = clean_source(inc_content)

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
    inc_header = get_header_raw(inc_content)

    def get_header(content):
        lines = content.split("\n")
        new_lines = []
        for line in lines:
            if "__kernel void" in line:
                break
            new_lines.append(line)
        for i in range(len(new_lines)):
            if "#include \"conv_kernel.inc.cl\"" in new_lines[i]:
                new_lines[i] = inc_header
        header = "\n".join(new_lines)
        new_lines = header.split("\n")
        for i in range(len(new_lines)):
            if "#include \"cl_common.h\"" in new_lines[i]:
                new_lines[i] = common_header
        header = "\n".join(new_lines)
        return header

    def get_funcs(content):
        funcs = {}
        lines = content.split("\n")
        first_kernel_idx = None
        for i in range(len(lines)):
            if "__kernel void" in lines[i]:
                first_kernel_idx = i
                break
        if first_kernel_idx is None:
            return funcs
        lines = lines[first_kernel_idx:]
        func = []
        name = ""
        for line in lines:
            if "__kernel void" in line:
                if name != "":
                    funcs[name] = "\n".join(func)
                    name = ""
                    func = []
                pattern = re.compile("__kernel void ([^(]+)\(")
                match = pattern.search(line)
                name = match.group(1)
            func.append(line)
        if name != "":
            funcs[name] = "\n".join(func)
            name = ""
            func = []
        return funcs

    filenames = os.listdir("cl_kernel")
    file_count = len(filenames)

    headers = {}
    funcs = {}
    for i in range(file_count):
        filename = filenames[i]
        infile = open("cl_kernel/" + filename, "r")
        content = infile.read()
        infile.close()
        content = clean_source(content)
        header = get_header(content)
        headers[filename] = header
        funcs_temp = get_funcs(content)
        for key in funcs_temp:
            funcs[key] = funcs_temp[key]

    core1 = ""
    core2 = ""

    for i in range(len(funcs)):
        func_name = list(funcs.keys())[i]
        content = funcs[func_name]
        if content == "":
            content = " "
        hexes = []
        for char in content:
            hexes.append(hex(ord(char)))
        core = "        {\"%s\", {" % func_name
        for item in hexes:
            core += str(item) + ", "
        core = core[: -2]
        core += "}}"
        if i != len(funcs) - 1:
            core += ",\n"
        core1 += core

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
        core = core[: -2]
        core += "}}"
        if i != len(headers) - 1:
            core += ",\n"
        core2 += core
    source = source % (core1, core2)
    print(source)

def gen_empty_opencl_kernels():
    source = """
    #pragma
    #ifdef PADDLE_MOBILE_CL
    #include <map>
    #include <string>
    #include <vector>
    namespace paddle_mobile {
        // func name => source
        extern const std::map<std::string, std::vector<unsigned char>> opencl_kernels = {
        };
        // file name => header
        extern const std::map<std::string, std::vector<unsigned char>> opencl_headers = {
        };
    }
    #endif
    """
    print(source)

if __name__ == "__main__":
    if sys.argv[1] == "0":
        gen_empty_opencl_kernels()
    elif sys.argv[1] == "1":
        gen_opencl_kernels()
