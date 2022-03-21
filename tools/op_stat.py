# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import re

def get_dirs(root_path):
  obj_list = os.listdir(root_path)
  for obj in obj_list:
    if os.path.isfile(os.path.join(root_path, obj)):
      obj_list.remove(obj)
  return obj_list

def get_kernels(root_path):
  kernel_path = "lite/kernels"
  kernel_suffix = "_compute.h"
  operator_suffix = "_op.cc"
  root = os.path.join(root_path, kernel_path)
  kernels_map = {}
  ops_map = {}

  devices = get_dirs(root)
  for device in devices:
      kernels_map[device] = []
      ops_map[device] = []
      for dirpath, dirnames, filenames in os.walk(os.path.join(root, device)):
        for f in filenames:
          kernels_map[device] = kernels_map[device] + re.findall(r"(.+?)" + kernel_suffix, f)
          ops_map[device] = ops_map[device] + re.findall(r"(.+?)" + operator_suffix, f)

  return kernels_map, ops_map

def suffix_filter(root, suffix):
  list = []
  for dirpath, dirnames, filenames in os.walk(root):
    for f in filenames:
      list = list + re.findall(r"(.+?)" + suffix, f)
  return list

def get_operators(root_path):
  operator_path = "lite/operators"
  operator_suffix = "_op.cc"
  root = os.path.join(root_path, operator_path)
  return suffix_filter(root, operator_suffix)

def get_passes(root_path):
  pass_path = "lite/core/optimizer/mir"
  pass_suffix = "_pass.cc"
  root = os.path.join(root_path, pass_path)
  return suffix_filter(root, pass_suffix)

def show_kernel_statistics(src_root):
  kernels_map, ops_map = get_kernels(src_root)
  print("=== Kernels Statistics === ")
  for device in kernels_map:
    kernel_num = len(kernels_map[device])
    op_num = len(ops_map[device])
    if kernel_num or op_num:
      print(device + " kernel: " + str(kernel_num) + "\t op: " + str(op_num))
  print("\n")

def show_operator_statistics(src_root):
  print("=== Operators Statistics === ")
  print("The number of operators is: " + str(len(get_operators(src_root))))
  print("\n")

def show_pass_statistics(src_root):
  print("=== Passes Statistics === ")
  print("The number of passes is: " + str(len(get_passes(src_root))))
  print("\n")

def main():
  path = os.path.join(os.path.join(__file__, os.pardir), os.pardir)
  src_root = os.path.abspath(path)
  show_kernel_statistics(src_root)
  show_operator_statistics(src_root)
  show_pass_statistics(src_root)

if __name__ == '__main__':
  main()
