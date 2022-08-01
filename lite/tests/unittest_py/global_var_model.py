# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import pickle
from pathlib import Path
import os

statics_data = {
    "targets": set(),
    "nnadapter_device_names": set(),
    "all_test_ops": {
        "Host": set(),
        "X86": set(),
        "ARM": set(),
        "OpenCL": set(),
        "Metal": set(),
        "XPU": set(),
        "NNAdapter": {
            "kunlunxin_xtcl": set(),
            "cambricon_mlu": set(),
            "nvidia_tensorrt": set(),
            "intel_openvino": set()
        }
    },
    "success_ops": {
        "Host": set(),
        "X86": set(),
        "ARM": set(),
        "OpenCL": set(),
        "Metal": set(),
        "XPU": set(),
        "NNAdapter": {
            "kunlunxin_xtcl": set(),
            "cambricon_mlu": set(),
            "nvidia_tensorrt": set(),
            "intel_openvino": set()
        }
    },
    "out_diff_ops": {
        "Host": set(),
        "X86": set(),
        "ARM": set(),
        "OpenCL": set(),
        "Metal": set(),
        "XPU": set(),
        "NNAdapter": {
            "kunlunxin_xtcl": set(),
            "cambricon_mlu": set(),
            "nvidia_tensorrt": set(),
            "intel_openvino": set()
        }
    },
    "lite_not_supported_ops": {
        "Host": set(),
        "X86": set(),
        "ARM": set(),
        "OpenCL": set(),
        "Metal": set(),
        "XPU": set(),
        "NNAdapter": {
            "kunlunxin_xtcl": set(),
            "cambricon_mlu": set(),
            "nvidia_tensorrt": set(),
            "intel_openvino": set()
        }
    },
    "paddle_not_supported_ops": {
        "Host": set(),
        "X86": set(),
        "ARM": set(),
        "OpenCL": set(),
        "Metal": set(),
        "XPU": set(),
        "NNAdapter": {
            "kunlunxin_xtcl": set(),
            "cambricon_mlu": set(),
            "nvidia_tensorrt": set(),
            "intel_openvino": set()
        }
    },
}
static_file = Path("./statics_data")
static_file_path_str = "./statics_data"


# coding=utf-8
def set_value(kind, target, nnadapter_device_name, op):
    if not static_file.exists():
        global statics_data
    else:
        with open(static_file_path_str, "rb") as f:
            statics_data = pickle.load(f)

    statics_data["targets"].add(target)
    if target == "NNAdapter":
        if nnadapter_device_name == "":
            print(
                "Args Error: Please set nnadapter_device_names when target=nnadapter!"
            )
            assert nnadapter_device_name != ""
        statics_data["nnadapter_device_names"].add(nnadapter_device_name)
        statics_data[kind][target][nnadapter_device_name].add(op)
    else:
        statics_data[kind][target].add(op)

    with open(static_file_path_str, "wb") as f:
        pickle.dump(statics_data, f)


def set_all_test_ops(target, nnadapter_device_name, op):
    set_value("all_test_ops", target, nnadapter_device_name, op)


def set_success_ops(target, nnadapter_device_name, op):
    set_value("success_ops", target, nnadapter_device_name, op)


def set_out_diff_ops(target, nnadapter_device_name, op):
    set_value("out_diff_ops", target, nnadapter_device_name, op)


def set_lite_not_supported_ops(target, nnadapter_device_name, op):
    set_value("lite_not_supported_ops", target, nnadapter_device_name, op)


def set_paddle_not_supported_ops(target, nnadapter_device_name, op):
    set_value("paddle_not_supported_ops", target, nnadapter_device_name, op)


def display():
    print("----------------------Unit Test Summary---------------------")
    with open("./statics_data", "rb") as f:
        statics_data = pickle.load(f)
        targets = statics_data["targets"]
        nnadapter_device_names = statics_data["nnadapter_device_names"]
    for target in targets:
        if target == "NNAdapter":
            display_nnadapter(statics_data, target, nnadapter_device_names)
            continue
        all_test_ops = statics_data["all_test_ops"][target]
        lite_not_supported_ops = statics_data["lite_not_supported_ops"][target]
        paddle_not_supported_ops = statics_data["paddle_not_supported_ops"][
            target]
        out_diff_ops = statics_data["out_diff_ops"][target]
        success_ops = statics_data["success_ops"][
            target] - lite_not_supported_ops - out_diff_ops - paddle_not_supported_ops

        print("Target =", target)
        print("Number of test =", len(all_test_ops))
        print("Number of success =", len(success_ops))
        print("Number of paddle not supported =",
              len(paddle_not_supported_ops))
        print("Number of lite not supported =", len(lite_not_supported_ops))
        print("Number of output diff =", len(out_diff_ops))
        print("\nDetails:")
        print("Success:")
        print(list(success_ops))
        print("\npaddle Not supported:")
        print(list(paddle_not_supported_ops))
        print("\nlite Not supported:")
        print(list(lite_not_supported_ops))
        print("\nOutput diff:")
        print(list(out_diff_ops))
        print("\n")


def display_nnadapter(statics_data, target, nnadapter_device_names):
    print("Target =", target)
    for nnadapter_device_name in nnadapter_device_names:
        if nnadapter_device_name == "":
            continue
        all_test_ops = statics_data["all_test_ops"][target][
            nnadapter_device_name]
        lite_not_supported_ops = statics_data["lite_not_supported_ops"][
            target][nnadapter_device_name]
        paddle_not_supported_ops = statics_data["paddle_not_supported_ops"][
            target][nnadapter_device_name]
        out_diff_ops = statics_data["out_diff_ops"][target][
            nnadapter_device_name]
        success_ops = statics_data["success_ops"][target][
            nnadapter_device_name] - lite_not_supported_ops - out_diff_ops - paddle_not_supported_ops

        print("nnadapter_device_name =", nnadapter_device_name)
        print("Number of test =", len(all_test_ops))
        print("Number of success =", len(success_ops))
        print("Number of paddle not supported =",
              len(paddle_not_supported_ops))
        print("Number of lite not supported =", len(lite_not_supported_ops))
        print("Number of output diff =", len(out_diff_ops))
        print("\nDetails:")
        print("Success:")
        print(list(success_ops))
        print("\npaddle Not supported:")
        print(list(paddle_not_supported_ops))
        print("\nlite Not supported:")
        print(list(lite_not_supported_ops))
        print("\nOutput diff:")
        print(list(out_diff_ops))
        print("\n")


if __name__ == "__main__":
    display()
