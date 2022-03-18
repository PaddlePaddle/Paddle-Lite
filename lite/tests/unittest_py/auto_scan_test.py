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

from auto_scan_base import IgnoreReasonsBase
import argparse
import platform
parser = argparse.ArgumentParser()
parser.add_argument("--target")
parser.add_argument(
    "--enforce_rpc", default='off', type=str, help="whther rpc is enforced")

parser.add_argument(
    "--server_ip",
    default="localhost",
    type=str,
    help="when rpc is used , the ip address of the server")

parser.add_argument(
    "--url",
    type=str,
    help="Address of model download in model test", )
parser.add_argument(
    "--file_name",
    type=str,
    help="File name of the compressed model package downloaded in the model test",
)
parser.add_argument(
    "--model_name",
    type=str,
    help="Model name(That is, the prefix of the compressed package) in model test",
)
parser.add_argument(
    "--input_shapes", help="The tested model's input_shapes", action="append")
parser.add_argument(
    "--nnadapter_device_names",
    default="",
    type=str,
    help="Set nnadapter device names")
parser.add_argument(
    "--nnadapter_context_properties",
    default="",
    type=str,
    help="Set nnadapter context properties")
parser.add_argument(
    "--nnadapter_model_cache_dir",
    default="",
    type=str,
    help="Set nnadapter model cache dir")
parser.add_argument(
    "--nnadapter_subgraph_partition_config_path",
    default="",
    type=str,
    help="Set nnadapter subgraph partition config path")
parser.add_argument(
    "--nnadapter_mixed_precision_quantization_config_path",
    default="",
    type=str,
    help="Set nnadapter mixed precision quantization config path")
args = parser.parse_args()

if (args.target == "ARM" and platform.system() == 'Darwin') or (
        args.target == "OpenCL") or (
            args.target == "Metal") or args.enforce_rpc == "on":
    from auto_scan_test_rpc import AutoScanTest
    from auto_scan_test_rpc import FusePassAutoScanTest
else:
    from auto_scan_test_no_rpc import AutoScanTest
    from auto_scan_test_no_rpc import FusePassAutoScanTest

IgnoreReasons = IgnoreReasonsBase
AutoScanTest = AutoScanTest
FusePassAutoScanTest = FusePassAutoScanTest
